"""
LESS denoising runner for GEVI movies.

Features
- Find input movies under ANALYSIS_ROOT using a path fragment with wildcards.
- Segment-wise denoising to limit RAM usage.
- Writes denoised H5 alongside input, plus begin/end illustration clips (AVI).
- Skips existing outputs by default, with an option to overwrite.

Notes on orientation
- Uses the (working) pre-bandpass behavior:
  1) Denoise segment
  2) Apply mask AFTER denoising (as requested)
  3) Flip denoised segment vertically: np.flip(..., axis=1)
"""

from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from less import denoise

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ----------------------------
# Config
# ----------------------------
ANALYSIS_ROOT = Path(r"Y:\Voltage\VisualConsciousness\Analysis\Visual")
INPUT_NAME = "cG_unmixedTR_dFF.h5"
OUTPUT_STEM = "cG_unmixedTR_dFF_denoised"  # output is OUTPUT_STEM + ".h5"


# ----------------------------
# File discovery
# ----------------------------
def iter_target_files(path_fragment: str):
    """
    Yield full paths to INPUT_NAME for each recording directory matched by path_fragment.

    Example fragments (relative to ANALYSIS_ROOT):
        r"cfm005mjr\\20251204\\meas03"
        r"cfm005mjr\\20251204\\meas0*"
        r"cfm005mjr\\202512*\\meas0*"
    """
    rel_pattern = str(Path(path_fragment) / INPUT_NAME)
    yield from (p for p in sorted(ANALYSIS_ROOT.glob(rel_pattern)) if p.is_file())


# ----------------------------
# H5 specs copying
# ----------------------------
def copy_specs_group(specs_in: h5py.Group, specs_out: h5py.Group) -> None:
    """
    Copy specs group attrs + datasets + extra_specs subgroup.
    Keeps the same structure as the input file.
    """
    for k, v in specs_in.attrs.items():
        specs_out.attrs[k] = v

    for k, v in specs_in.items():
        if isinstance(v, h5py.Dataset):
            specs_out.create_dataset(k, data=v[()])

    if "extra_specs" in specs_in and isinstance(specs_in["extra_specs"], h5py.Group):
        extra_in = specs_in["extra_specs"]
        extra_out = specs_out.create_group("extra_specs")
        for k, v in extra_in.items():
            if isinstance(v, h5py.Dataset):
                extra_out.create_dataset(k, data=v[()])


# ----------------------------
# Preprocessing helpers
# ----------------------------
def mask_movie(movie: np.ndarray, raw_mask: np.ndarray, binning: int) -> np.ndarray:
    """
    Resize the raw (unbinned) mask down to movie resolution, crop to (H, W), then apply.

    movie: (T, H, W)
    raw_mask: mask stored in specs (often bool/int, sometimes has extra singleton dims)
    binning: spatial binning factor
    """
    if binning is None or int(binning) <= 0:
        raise ValueError(f"Invalid binning: {binning}")

    raw_mask = np.asarray(raw_mask)
    while raw_mask.ndim > 2:
        raw_mask = raw_mask[0]
    if raw_mask.ndim != 2:
        raise ValueError(f"Expected raw_mask to be 2D after squeezing; got shape {raw_mask.shape}")

    # OpenCV is happiest with numeric contiguous arrays
    raw_mask_f32 = np.ascontiguousarray(raw_mask.astype(np.float32))
    scale = 1.0 / float(binning)

    mask_resized = cv2.resize(
        raw_mask_f32,
        dsize=(0, 0),
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_LINEAR,
    )

    _, H, W = movie.shape
    mask_resized = mask_resized[:H, :W]
    mask_bool = mask_resized > 0.5

    return movie * mask_bool


def run_less_denoising(movie_np: np.ndarray, *, cuda: bool = True, **kwargs) -> np.ndarray:
    """
    Run LESS denoising on a (T, H, W) float movie and return float32 numpy output.
    """
    if movie_np.ndim != 3:
        raise ValueError(f"Expected movie with shape (T, H, W); got {movie_np.shape}")

    movie_tensor = torch.from_numpy(movie_np).float()
    out_tensor = denoise(data=movie_tensor, cuda=cuda, **kwargs)
    return out_tensor.detach().cpu().numpy().astype(np.float32, copy=False)


# ----------------------------
# Illustration video helpers
# ----------------------------
def _to_bgr_frames_diverging(
    clip: np.ndarray,
    *,
    vlim: float | None = None,
    cmap_name: str = "seismic",
) -> np.ndarray:
    """
    Convert a float clip (T, H, W) to BGR uint8 frames (T, H, W, 3) using a diverging colormap.

    Uses symmetric limits about 0:
      - vlim defaults to 99th percentile of |clip| (robust to outliers)
    """
    clip = np.asarray(clip, dtype=np.float32)
    np.nan_to_num(clip, copy=False)

    if vlim is None:
        vlim = float(np.percentile(np.abs(clip), 99))
        if not np.isfinite(vlim) or vlim <= 0:
            vmax = float(np.max(np.abs(clip)))
            vlim = vmax if vmax > 0 else 1.0

    norm = mcolors.TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
    cmap = plt.get_cmap(cmap_name)

    frames_bgr = []
    for frame in clip:
        rgba = cmap(norm(frame))  # (H, W, 4), floats in [0, 1]
        rgb = (rgba[..., :3] * 255).astype(np.uint8)
        frames_bgr.append(rgb[..., ::-1])  # RGB -> BGR

    return np.stack(frames_bgr, axis=0)


def save_begin_end_clips(
    h5_path: Path,
    *,
    out_dirname: str = "illustrations",
    seconds: float = 5.0,
    cmap_name: str = "seismic",
) -> tuple[Path, Path]:
    """
    Save AVI (MJPG) clips of the first and last `seconds` from the movie stored in `h5_path`.

    Filenames:
      - cG_unmixedTR_dFF_denoised_begin.avi
      - cG_unmixedTR_dFF_denoised_end.avi
    """
    out_dir = h5_path.parent / out_dirname
    out_dir.mkdir(parents=True, exist_ok=True)

    begin_path = out_dir / f"{OUTPUT_STEM}_begin.avi"
    end_path = out_dir / f"{OUTPUT_STEM}_end.avi"

    with h5py.File(h5_path, "r") as f:
        mov = f["mov"]
        fps = float(np.array(f["specs"]["fps"][()]).squeeze())
        T, H, W = mov.shape

        n = int(round(seconds * fps))
        n = max(1, min(n, T))

        begin = np.array(mov[0:n], dtype=np.float32)
        end = np.array(mov[T - n : T], dtype=np.float32)

    begin_bgr = _to_bgr_frames_diverging(begin, cmap_name=cmap_name)
    end_bgr = _to_bgr_frames_diverging(end, cmap_name=cmap_name)

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    def _write_avi(path: Path, frames_bgr: np.ndarray):
        vw = cv2.VideoWriter(str(path), fourcc, fps/5, (W, H), isColor=True)
        if not vw.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for {path}")
        try:
            for frame in frames_bgr:
                vw.write(frame)
        finally:
            vw.release()

    _write_avi(begin_path, begin_bgr)
    _write_avi(end_path, end_bgr)

    return begin_path, end_path


# ----------------------------
# Core processing
# ----------------------------
def denoise_movie_file(
    input_path: Path,
    *,
    skip: bool = True,
    segment_length: int = 5000,
    denoise_kwargs: dict | None = None,
    cuda: bool = True,
) -> Path:
    """
    Denoise a movie file in segments and write output next to input.

    Orientation behavior:
      - Denoise segment
      - Apply mask after denoising
      - Flip denoised segment vertically: np.flip(..., axis=1)

    Parameters
    - input_path: path to INPUT_NAME
    - skip: if True, keep existing outputs; if False, overwrite
    - segment_length: frames per denoising segment
    - denoise_kwargs: passed to less.denoise (e.g., {'patch_size': 16, 'verbose': False})
    - cuda: pass to LESS (cuda=True uses GPU)
    """
    denoise_kwargs = denoise_kwargs or {"patch_size": 16, "verbose": False}

    output_path = input_path.with_name(f"{OUTPUT_STEM}.h5")

    # H5 overwrite/skip logic
    if output_path.exists():
        if skip:
            print(f"    Skipping denoise (exists): {output_path.name}", flush=True)
        else:
            output_path.unlink()

    if not output_path.exists():
        with h5py.File(input_path, "r") as f_in:
            mov_in = f_in["mov"]
            specs_in = f_in["specs"]

            binning = int(np.array(specs_in["binning"][()]).squeeze())
            raw_mask = specs_in["extra_specs"]["mask"][()]

            T, H, W = mov_in.shape
            n_segments = (T + segment_length - 1) // segment_length

            with h5py.File(output_path, "w") as f_out:
                mov_out = f_out.create_dataset(
                    "mov",
                    shape=(T, H, W),
                    dtype="float32",
                    compression="gzip",
                )

                specs_out = f_out.create_group("specs")
                copy_specs_group(specs_in, specs_out)

                for seg_idx, start in enumerate(range(0, T, segment_length), start=1):
                    end = min(start + segment_length, T)
                    print(f"    Segment {seg_idx}/{n_segments} (frames {start}:{end})", flush=True)

                    segment = np.array(mov_in[start:end], dtype=np.float32)
                    np.nan_to_num(segment, copy=False)

                    # Denoise first
                    segment_denoised = run_less_denoising(segment, cuda=cuda, **denoise_kwargs)

                    # Mask after denoising (your preferred behavior)
                    segment_denoised = mask_movie(segment_denoised, raw_mask, binning)

                    # Flip vertically (axis=1), consistent with the previously working version
                    segment_denoised = np.flip(segment_denoised, axis=1)

                    mov_out[start:end] = segment_denoised

    # Clips overwrite/skip logic
    ill_dir = output_path.parent / "illustrations"
    begin_clip = ill_dir / f"{OUTPUT_STEM}_begin.avi"
    end_clip = ill_dir / f"{OUTPUT_STEM}_end.avi"

    if begin_clip.exists() and end_clip.exists() and skip:
        print("    Clips exist; skipping.", flush=True)
    else:
        if not skip:
            if begin_clip.exists():
                begin_clip.unlink()
            if end_clip.exists():
                end_clip.unlink()

        print("    Writing begin/end clips to illustrations/ ...", flush=True)
        bpath, epath = save_begin_end_clips(output_path)
        print(f"    Saved clips: {bpath.name}, {epath.name}", flush=True)

    return output_path


def main(
    path_fragment: str,
    *,
    skip: bool = True,
    segment_length: int = 5000,
    denoise_kwargs: dict | None = None,
    cuda: bool = True,
):
    """
    Batch denoise movies matched by `path_fragment`.

    Parameters
    - path_fragment: relative path under ANALYSIS_ROOT, may include '*' wildcards.
    - skip: if True, keep existing outputs; if False, overwrite existing outputs.
    - segment_length: frames per denoising segment.
    - denoise_kwargs: passed to less.denoise (e.g., {'patch_size': 16, 'verbose': False})
    - cuda: pass to LESS (cuda=True uses GPU)
    """
    denoise_kwargs = denoise_kwargs or {"patch_size": 16, "verbose": False}

    targets = list(iter_target_files(path_fragment))
    if not targets:
        print(f"No files found for: {path_fragment}")
        return

    print(f"Found {len(targets)} file(s). skip={skip}.")
    for i, input_path in enumerate(targets, start=1):
        print(f"[{i}/{len(targets)}] Processing: {input_path}", flush=True)
        try:
            out = denoise_movie_file(
                input_path,
                skip=skip,
                segment_length=segment_length,
                denoise_kwargs=denoise_kwargs,
                cuda=cuda,
            )
            print(f"    Output: {out}", flush=True)
        except Exception as e:
            print(f"    ERROR: {e}", flush=True)
            continue

    print("Complete!")


# Example usage:
# main(r"cmm005mjr\20251202\meas04", skip=True)
main(r"cmm005mjr/20251202/meas04", skip=False)
#main(r"cmm005mjr/20251127/meas0*", skip=False)