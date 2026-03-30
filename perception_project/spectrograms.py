"""
Precompute and save per-trial mean-trace wavelet spectrograms for multiple brain regions
(per recording) to Zarr.

For ONE mouse:
- Reads trial info from: trial_info/TrialInfo_<mouse>.csv  (relative to this script)
- Finds recordings under:
    Y:/Voltage/VisualConsciousness/Analysis/Visual/<mouse>/<YYYYMMDD>/measXX\
- For each recording folder:
    - default: SKIP if trial_spectrograms.zarr already exists
    - optional: OVERWRITE if --overwrite is passed

Per recording it will:
    - use movie: cG_unmixedTR_dFF.h5
    - load metadata + masks once (no full movie load)
    - apply brain+vessel mask from specs["extra_specs"]["mask"]
    - compute region masks from Allen outlines:
        V1    -> index 37
        M1    -> index 3
        VISa  -> index 53
        VISam -> index 33
        VISpm -> index 41
      then combined_mask = brain_mask & region_mask
    - per trial, load only the needed clip from HDF5, flip vertically, compute mean traces
      over combined_mask pixels for each region
    - align to STIMULUS OFFSET, using:
        offset_time = BFMTime + Duration
      and extract window:
        [offset_time - 3, offset_time + 2] seconds
    - compute spectrograms for each trial & region using spectrogram() (unchanged)
    - save to: trial_spectrograms.zarr in the recording folder

Zarr contents:
- S            float32  (n_trials, n_regions, n_freq, n_time)   (freq x time)
- MeanTrace    float32  (n_trials, n_regions, n_time)
- trial_id     string   (n_trials,)
- trial_type   string   (n_trials,)  from CSV column "TrialType"
- region_name  string   (n_regions,) ["V1","M1","VISa","VISam","VISpm"]
- freqs        float32  (n_freq,)
- time         float32  (n_time,)    seconds relative to OFFSET (t=0 is offset)

Run:
    python precompute_trial_spectrograms.py cmm005mjr
    python precompute_trial_spectrograms.py cmm005mjr --overwrite
"""

from __future__ import annotations

import sys
import shutil
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import cv2
import pywt
import zarr
from numcodecs import Blosc, VLenUTF8
from skimage.draw import polygon2mask


# =========================
# DO NOT CHANGE (per user)
# =========================
def spectrogram(segment, fps, f_range=(2, 60), num_freqs=100):
    """
    Compute a wavelet spectrogram of a signal segment using MATLAB-matching parameters.
    
    Parameters:
        segment: 1D numpy array (signal trace for a single trial)
        fps: frames per second (sampling rate)
        f_range: frequency range (Hz)
        num_freqs: number of frequency bins (log spaced)
        time_bandwidth: MATLAB-like 'TimeBandwidth' parameter
        voices_per_octave: MATLAB-like 'VoicesPerOctave' parameter

    Returns:
        power: normalized wavelet magnitude (freq x time)
        freqs: corresponding frequencies in Hz
    """
    dt = 1 / fps
    freqs = np.logspace(np.log10(f_range[0]), np.log10(f_range[1]), num_freqs)

    # Morelet wavelet
    wavelet = "cmor2.0-1.0"

    # Match MATLAB scale construction
    scales = pywt.scale2frequency(wavelet, 1.0) / freqs * fps

    # Compute wavelet transform
    coeffs, actual_freqs = pywt.cwt(segment, scales, wavelet, sampling_period=dt)

    # Normalize as MATLAB: abs(wt) / sqrt(freq)
    amplitude = np.abs(coeffs) / np.sqrt(actual_freqs[:, np.newaxis])

    return amplitude, actual_freqs


# =========================
# Region specification
# =========================
# NOTE: dict insertion order is preserved in Python 3.7+, so region order is stable.
REGION_SPECS = {
    "V1": [37],
    "M1": [3],
    "VISa": [53],
    "VISam": [33],
    "VISpm": [41],
}


# =========================
# Masking helpers
# =========================
def resize_and_flip_brain_mask(raw_mask, binning, movie_hw):
    """
    Convert specs["extra_specs"]["mask"] into a boolean mask aligned to the movie after movie flip.

    Your mask_movie() does:
      - resize
      - crop
      - flip axis 0

    We'll replicate that here so we can form a combined mask for mean-trace averaging.
    """
    raw_mask = np.asarray(raw_mask).astype(np.float32)
    mask = cv2.resize(
        raw_mask, (0, 0), fx=1 / binning, fy=1 / binning, interpolation=cv2.INTER_LINEAR
    )
    mask = mask[: movie_hw[0], : movie_hw[1]]
    mask = np.flip(mask, axis=0)
    return mask.astype(bool)


def region_mask_from_indices(raw_outlines, binning, spaceorigin, movie_shape_hw, indices):
    """
    Create a boolean mask by taking the union of the polygon masks for the provided outline indices.
    Matches the outline logic you provided.
    """
    if indices is None or len(indices) == 0:
        raise ValueError("indices must be a non-empty list of outline indices")

    spaceorigin = (spaceorigin - 1) / binning + 1

    outlines_nums = np.arange(raw_outlines.shape[2])
    outlines = raw_outlines[:, :, outlines_nums] / binning

    outlines[:, 0, :] -= spaceorigin[1] - 1
    outlines[:, 1, :] -= spaceorigin[0] - 1

    total_mask = np.zeros(movie_shape_hw, dtype=bool)

    for i in indices:
        outline = outlines[i, :, :]
        valid = ~np.isnan(outline).any(axis=0)
        x_coords = outline[1, valid]
        y_coords = outline[0, valid]

        roi_mask = polygon2mask(movie_shape_hw, np.column_stack((y_coords, x_coords)))
        roi_mask = np.flipud(roi_mask).astype(bool)
        total_mask |= roi_mask

    return total_mask


def load_masks_and_metadata(path_movie: str, region_specs: dict[str, list[int]]):
    """
    Loads ONLY metadata needed to build combined masks + fps, without loading the full movie.

    Returns:
        combined_masks: dict region_name -> (H,W) bool  (brain/vessel mask & region mask)
        fps: float
        T: int number of frames
    """
    with h5py.File(path_movie, "r") as f:
        dset = f["mov"]
        T, H, W = dset.shape

        specs = f["specs"]
        fps = float(specs["fps"][()].squeeze())
        raw_mask = specs["extra_specs"]["mask"][()].squeeze()
        binning = float(specs["binning"][()].squeeze())
        raw_outlines = specs["extra_specs"]["allenMapEdgeOutline"][()].squeeze()
        spaceorigin = specs["spaceorigin"][()].squeeze()

    brain_mask = resize_and_flip_brain_mask(raw_mask, binning, (H, W))

    combined_masks = {}
    for rname, idxs in region_specs.items():
        rmask = region_mask_from_indices(raw_outlines, binning, spaceorigin, (H, W), idxs)
        combined = brain_mask & rmask
        combined_masks[rname] = combined

    return combined_masks, fps, int(T)


# =========================
# Recording discovery / CSV filtering
# =========================
def _match_date_rows(df: pd.DataFrame, date_folder: str) -> pd.Series:
    """
    date_folder: "YYYYMMDD"
    trial CSV Date often looks like YYMMDD (int) or string.
    We match by comparing the last 6 digits.
    """
    target = date_folder[2:]  # "YYMMDD"
    s = df["Date"].astype(str).str.replace(".0", "", regex=False).str.zfill(6)
    return s == target


def _canon_meas(x) -> str:
    """
    Canonicalize a 'File' field to 'measXX' lowercase where possible.
    Handles:
      - 'meas03', 'meas03/', '20251208/meas03'
      - numeric 3 or '03' -> 'meas03'
      - strips whitespace
    """
    if pd.isna(x):
        return ""
    s = str(x).strip().replace("\\", "/")
    if "/" in s:
        s = s.split("/")[-1].strip()
    s = s.strip()

    if s.lower().startswith("meas"):
        tail = s[4:].strip()
        if tail.isdigit():
            return f"meas{int(tail):02d}"
        return s.lower()

    if s.isdigit():
        return f"meas{int(s):02d}"

    return s.lower()


def find_recording_folders(base_analysis: Path, mouse: str) -> list[Path]:
    """
    Returns list of recording folders like:
    base_analysis/mouse/YYYYMMDD/measXX
    """
    mouse_dir = base_analysis / mouse
    if not mouse_dir.exists():
        print(f"[WARN] Mouse directory does not exist: {mouse_dir}")
        return []

    recs = []
    for date_dir in sorted(mouse_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        name = date_dir.name
        if len(name) != 8 or not name.isdigit():
            continue
        for meas_dir in sorted(date_dir.iterdir()):
            if meas_dir.is_dir() and meas_dir.name.lower().startswith("meas"):
                recs.append(meas_dir)

    return recs


# =========================
# Core processing
# =========================
def compute_and_save_recording(
    rec_dir: Path,
    df_mouse: pd.DataFrame,
    overwrite: bool = False,
    pre_s: float = 3.0,      # pre-offset
    post_s: float = 2.0,     # post-offset
    f_range=(2, 60),
    num_freqs: int = 100,
    print_every_trials: int = 25,
):
    """
    Process ONE recording folder and save trial_spectrograms.zarr inside it.
    Default behavior: skip if file exists.
    If overwrite=True: delete existing trial_spectrograms.zarr and recompute.

    Uses per-trial slicing to avoid loading the full movie into RAM.
    Aligns to stimulus OFFSET using BFMTime + Duration, window [-pre_s, +post_s].
    Saves multi-region spectrograms + mean traces.
    """
    date_folder = rec_dir.parent.name  # YYYYMMDD
    meas_folder = rec_dir.name         # measXX
    meas_canon = meas_folder.lower()

    out_zarr = rec_dir / "trial_spectrograms.zarr"
    if out_zarr.exists():
        if not overwrite:
            print(f"[SKIP] Already exists: {out_zarr}")
            return
        print(f"[OVERWRITE] Deleting existing: {out_zarr}")
        try:
            shutil.rmtree(out_zarr)
        except Exception as e:
            print(f"[ERROR] Could not delete existing zarr store: {out_zarr} ({e})")
            traceback.print_exc()
            return

    movie_path = rec_dir / "cG_unmixedTR_dFF.h5"
    if not movie_path.exists():
        print(f"[SKIP] Movie not found: {movie_path}")
        return

    # Filter trials for this recording
    try:
        df_day = df_mouse[_match_date_rows(df_mouse, date_folder)]
        df_rec = df_day[df_day["File_canon"] == meas_canon].copy()
    except Exception:
        print(f"[ERROR] Could not filter trials for {date_folder}/{meas_folder}")
        traceback.print_exc()
        return

    if len(df_rec) == 0:
        print(f"[SKIP] No trials found in CSV for {date_folder}/{meas_folder}")
        return

    # Diagnostic + de-dup by TrialID
    n_rows_raw = len(df_rec)
    n_unique_raw = df_rec["TrialID"].nunique(dropna=False)
    n_dupes = n_rows_raw - n_unique_raw

    if n_dupes > 0:
        dup_ids = (
            df_rec.loc[df_rec.duplicated(subset=["TrialID"], keep=False), "TrialID"]
            .astype(str)
            .unique()
        )
        print(f"[WARN] Found {n_dupes} duplicate rows by TrialID in {date_folder}/{meas_folder}. Dropping duplicates.")
        print(f"       Example duplicated TrialIDs (up to 10): {list(dup_ids[:10])}")

    df_rec = df_rec.drop_duplicates(subset=["TrialID"], keep="first").copy()

    print(f"\n[REC] {date_folder}/{meas_folder}")
    print(f"      Trials in CSV (raw rows): {n_rows_raw}")
    print(f"      Trials after de-dup by TrialID: {len(df_rec)}")
    print(f"      Using movie: {movie_path}")

    # Load masks + fps without loading full movie
    try:
        combined_masks, fps, T = load_masks_and_metadata(str(movie_path), REGION_SPECS)
    except Exception as e:
        print(f"[ERROR] Failed to load masks/metadata for {date_folder}/{meas_folder}: {e}")
        traceback.print_exc()
        return

    region_names = list(REGION_SPECS.keys())
    n_regions = len(region_names)

    # Precompute per-region pixel indices and validate
    region_pix = {}
    for rname in region_names:
        pix = np.where(combined_masks[rname])
        if len(pix[0]) == 0:
            print(f"[ERROR] Region {rname} has 0 pixels after masking in {date_folder}/{meas_folder}. Treating as bug; skipping recording.")
            return
        region_pix[rname] = pix

    # Precompute time axis (relative to OFFSET)
    n_t_expected = int(round((pre_s + post_s) * fps))
    time_arr = (np.arange(n_t_expected, dtype=np.float32) / fps) - np.float32(pre_s)

    print(f"      fps={fps:.3f}, T={T} frames")
    for rname in region_names:
        print(f"      masked pixels ({rname}) = {len(region_pix[rname][0])}")
    print(f"      OFFSET-aligned window=[-{pre_s}, +{post_s}] s -> {n_t_expected} frames")
    print(f"      spectrogram: f_range={f_range}, num_freqs={num_freqs}")
    print("      Loading per-trial clips from HDF5 (not loading full movie).")

    all_S = []
    all_mean = []
    all_trial_ids = []
    all_trial_types = []
    omitted_trials = []
    freqs_ref = None

    # Sort by TrialID for reproducibility
    try:
        df_rec = df_rec.sort_values("TrialID")
    except Exception:
        pass

    # Open file once per recording for efficient repeated slicing
    try:
        f = h5py.File(str(movie_path), "r")
        dset = f["mov"]

        for k, (_, trial) in enumerate(df_rec.iterrows(), start=1):
            trial_id = trial.get("TrialID", None)
            bfm_time = trial.get("BFMTime", None)
            duration = trial.get("Duration", None)

            if bfm_time is None or (isinstance(bfm_time, float) and np.isnan(bfm_time)):
                omitted_trials.append(trial_id)
                print(f"        [TRIAL {k}/{len(df_rec)}] Skip TrialID={trial_id}: missing BFMTime")
                continue
            if duration is None or (isinstance(duration, float) and np.isnan(duration)):
                omitted_trials.append(trial_id)
                print(f"        [TRIAL {k}/{len(df_rec)}] Skip TrialID={trial_id}: missing Duration")
                continue

            try:
                offset_time = float(bfm_time) + float(duration)

                start_frame = int(round((offset_time - pre_s) * fps))
                end_frame = start_frame + n_t_expected

                if start_frame < 0 or end_frame > T:
                    omitted_trials.append(trial_id)
                    print(f"        [TRIAL {k}/{len(df_rec)}] Skip TrialID={trial_id}: out of bounds (start={start_frame}, end={end_frame}, T={T})")
                    continue

                # Load only the frames needed for this trial
                clip = dset[start_frame:end_frame, :, :]  # (t,h,w)

                # Convert + handle NaNs
                clip = np.nan_to_num(clip).astype(np.float32)

                # Vertical flip to match convention (same as before)
                clip = np.flip(clip, axis=1)

                # Compute mean traces for each region
                mean_mat = np.empty((n_regions, n_t_expected), dtype=np.float32)
                S_mat = None  # (n_regions, n_freq, n_time)

                for ri, rname in enumerate(region_names):
                    pix = region_pix[rname]
                    mean_signal = clip[:, pix[0], pix[1]].mean(axis=1)  # (t,)
                    mean_mat[ri, :] = mean_signal.astype(np.float32)

                    S, freqs = spectrogram(mean_signal, fps, f_range=f_range, num_freqs=num_freqs)

                    # Enforce time length consistency
                    if S.shape[1] != n_t_expected:
                        min_len = min(S.shape[1], n_t_expected)
                        S = S[:, :min_len]
                        if min_len < n_t_expected:
                            pad = np.full((S.shape[0], n_t_expected - min_len), np.nan, dtype=S.dtype)
                            S = np.concatenate([S, pad], axis=1)

                    if freqs_ref is None:
                        freqs_ref = freqs.astype(np.float32)
                    else:
                        if freqs.shape != freqs_ref.shape or np.max(np.abs(freqs.astype(np.float32) - freqs_ref)) > 1e-3:
                            print(f"        [WARN] Frequency array mismatch (region {rname}) TrialID={trial_id}. Using freqs from first computed trial.")

                    if S_mat is None:
                        S_mat = np.empty((n_regions, S.shape[0], S.shape[1]), dtype=np.float32)

                    S_mat[ri, :, :] = S.astype(np.float32)

                all_S.append(S_mat)
                all_mean.append(mean_mat)
                all_trial_ids.append(str(trial_id) if trial_id is not None else "NA")
                all_trial_types.append(str(trial.get("TrialType", "NA")))

                if (k % print_every_trials) == 0 or k == 1 or k == len(df_rec):
                    print(f"        Progress: {k}/{len(df_rec)} trials processed (saved: {len(all_S)}, omitted: {len(omitted_trials)})")

            except Exception as e:
                omitted_trials.append(trial_id)
                print(f"        [TRIAL {k}/{len(df_rec)}] ERROR TrialID={trial_id}: {e}")
                traceback.print_exc()
                continue

    except Exception as e:
        print(f"[ERROR] Failed while reading clips for {date_folder}/{meas_folder}: {e}")
        traceback.print_exc()
        return
    finally:
        try:
            f.close()
        except Exception:
            pass

    if len(all_S) == 0:
        print(f"[WARN] No spectrograms computed successfully for {date_folder}/{meas_folder}. Nothing saved.")
        return

    S_arr = np.stack(all_S, axis=0).astype(np.float32)         # (trial, region, freq, time)
    mean_arr = np.stack(all_mean, axis=0).astype(np.float32)   # (trial, region, time)
    freqs_ref = freqs_ref if freqs_ref is not None else np.array([], dtype=np.float32)

    print(f"      Done computing.")
    print(f"      S shape: {S_arr.shape} (trial, region, freq, time)")
    print(f"      MeanTrace shape: {mean_arr.shape} (trial, region, time)")
    print(f"      Saving Zarr to: {out_zarr}")

    try:
        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
        root = zarr.open_group(str(out_zarr), mode="w")

        chunk_trials = 1 if S_arr.shape[0] < 8 else 8
        chunks_S = (chunk_trials, 1, S_arr.shape[2], S_arr.shape[3])
        chunks_M = (chunk_trials, 1, mean_arr.shape[2])

        root.create_dataset(
            name="S",
            data=S_arr,
            chunks=chunks_S,
            dtype="float32",
            compressor=compressor,
            overwrite=True,
        )

        root.create_dataset(
            name="MeanTrace",
            data=mean_arr,
            chunks=chunks_M,
            dtype="float32",
            compressor=compressor,
            overwrite=True,
        )

        root.create_dataset(
            name="freqs",
            data=freqs_ref.astype(np.float32),
            dtype="float32",
            compressor=compressor,
            overwrite=True,
        )

        root.create_dataset(
            name="time",
            data=time_arr.astype(np.float32),
            dtype="float32",
            compressor=compressor,
            overwrite=True,
        )

        utf8 = VLenUTF8()

        root.create_dataset(
            name="trial_id",
            data=np.array(all_trial_ids, dtype=object),
            dtype=object,
            object_codec=utf8,
            overwrite=True,
        )

        root.create_dataset(
            name="trial_type",
            data=np.array(all_trial_types, dtype=object),
            dtype=object,
            object_codec=utf8,
            overwrite=True,
        )

        root.create_dataset(
            name="region_name",
            data=np.array(region_names, dtype=object),
            dtype=object,
            object_codec=utf8,
            overwrite=True,
        )

        root.attrs["spectrogram_version"] = "multi_region_v1"
        root.attrs["align_to"] = "offset"
        root.attrs["alignment_fields"] = ["BFMTime", "Duration"]

        root.attrs["mouse"] = str(df_mouse["AnimalCode"].iloc[0]) if "AnimalCode" in df_mouse.columns and len(df_mouse) > 0 else "NA"
        root.attrs["date_folder"] = date_folder
        root.attrs["meas_folder"] = meas_folder
        root.attrs["movie_path"] = str(movie_path)
        root.attrs["fps"] = float(fps)
        root.attrs["pre_s"] = float(pre_s)
        root.attrs["post_s"] = float(post_s)
        root.attrs["f_range"] = list(map(float, f_range))
        root.attrs["num_freqs"] = int(num_freqs)

        root.attrs["regions"] = region_names
        root.attrs["region_outline_indices"] = {k: v for k, v in REGION_SPECS.items()}

        root.attrs["n_trials_saved"] = int(S_arr.shape[0])
        root.attrs["n_trials_in_csv_raw_rows"] = int(n_rows_raw)
        root.attrs["n_trials_after_dedup_trialid"] = int(len(df_rec))
        root.attrs["n_trials_omitted"] = int(len(omitted_trials))

        if len(omitted_trials) > 0:
            root.attrs["omitted_trial_ids"] = [str(x) for x in omitted_trials[:500]]

        print(f"      Saved: {out_zarr}")
        print(f"      Saved trials: {S_arr.shape[0]} | Omitted trials: {len(omitted_trials)}")

    except Exception as e:
        print(f"[ERROR] Failed to save Zarr for {date_folder}/{meas_folder}: {e}")
        traceback.print_exc()
        return


def run_mouse(
    mouse: str,
    overwrite: bool = False,
    base_analysis: str = r"Y:/Voltage/VisualConsciousness/Analysis/Visual",
    f_range=(2, 60),
    num_freqs: int = 100,
):
    """
    Main entry for one mouse.
    """
    script_dir = Path(__file__).resolve().parent
    trial_info_path = script_dir / "trial_info" / f"TrialInfo_{mouse}.csv"

    print(f"\n[MOUSE] {mouse}")
    print(f"       Trial info: {trial_info_path}")
    print(f"       Analysis base: {base_analysis}")
    print(f"       overwrite={overwrite}")

    if not trial_info_path.exists():
        raise FileNotFoundError(f"Trial info CSV not found: {trial_info_path}")

    df_all = pd.read_csv(trial_info_path)

    # Normalize key columns
    if "AnimalCode" in df_all.columns:
        df_all["AnimalCode"] = df_all["AnimalCode"].astype(str).str.strip()

    if "File" in df_all.columns:
        df_all["File_canon"] = df_all["File"].apply(_canon_meas)
    else:
        raise ValueError("Trial info CSV missing required column: File")

    # Filter to this mouse if possible
    if "AnimalCode" in df_all.columns:
        df_mouse = df_all[df_all["AnimalCode"].astype(str) == str(mouse)].copy()
    else:
        print("[WARN] CSV missing 'AnimalCode' column. Using full CSV as df_mouse.")
        df_mouse = df_all.copy()

    required_cols = ["Date", "File", "TrialID", "BFMTime", "Duration", "TrialType"]
    missing = [c for c in required_cols if c not in df_mouse.columns]
    if missing:
        raise ValueError(f"Trial info CSV is missing required columns: {missing}")

    recs = find_recording_folders(Path(base_analysis), mouse)
    print(f"       Found recording folders: {len(recs)}")

    if overwrite:
        n_to_run = len(recs)
    else:
        n_to_run = sum(
            1 for rec_dir in recs if not (rec_dir / "trial_spectrograms.zarr").exists()
        )
    print(f"       Recordings to process: {n_to_run}")

    i = 0
    for rec_dir in recs:
        out_zarr = rec_dir / "trial_spectrograms.zarr"
        if out_zarr.exists() and not overwrite:
            continue

        i += 1
        print(f"\n[QUEUE] Processing recording {i}/{n_to_run}: {rec_dir.parent.name}/{rec_dir.name}")

        try:
            compute_and_save_recording(
                rec_dir=rec_dir,
                df_mouse=df_mouse,
                overwrite=overwrite,
                pre_s=3.0,
                post_s=2.0,
                f_range=f_range,
                num_freqs=num_freqs,
            )
        except Exception as e:
            print(f"[ERROR] Recording-level failure for {rec_dir}: {e}")
            traceback.print_exc()
            print("[CONTINUE] Moving to next recording.")
            continue

    print(f"\n[DONE] Finished mouse {mouse}.")


def _parse_args(argv: list[str]) -> tuple[str, bool]:
    """
    Minimal CLI parsing:
      python script.py <mouse> [--overwrite]
    """
    if len(argv) < 2:
        print("Usage: python precompute_trial_spectrograms.py <mouse> [--overwrite]")
        sys.exit(1)

    mouse = argv[1]
    overwrite = "--overwrite" in argv[2:]
    return mouse, overwrite


if __name__ == "__main__":
    mouse_name, overwrite_flag = _parse_args(sys.argv)
    run_mouse(mouse_name, overwrite=overwrite_flag)