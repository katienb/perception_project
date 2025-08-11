# Task (update v2): Single-file implementation of “band-limited power maps”

*Folder:* `wave_detection/`  
*Output script name:* `band_power_maps.py`  
*Reference loader:* reuse the simple movie-loading code that already works in `visualization.ipynb`.  
*Purpose:* visualise band-limited power in cortical GEVI movies to reveal candidate travelling waves.

---

## Workflow Claude should follow

### 1 Command-line interface  
* Accept the following **arguments / options** (with reasonable defaults):  
  * `--movie_path`  (required) – path to a single movie file.  
  * `--t_start` and `--t_stop`  (floats, seconds; default 202 and 204) – crop window.  
  * `--band_low` and `--band_high`  (floats, Hz; default something like 30 70).  
  * `--display`  (flag) – show figures onscreen (always true for this assignment).  
  * `--save`  (flag) – if the user adds this flag, also write the figures to disk; otherwise do **not** save.  
  * `--outdir`  (str; default `wave_detection/results/`) – only used when `--save` is present.

### 2 Data loading & cropping  
1. Import the loader from `visualization.ipynb` (or paste its code).  
2. The loader must return:  
   * `movie` – NumPy array shaped `(T, Y, X)` in the movie’s original units (no normalisation yet).  
   * `fps` – frame-rate in Hz.  
3. Convert the start/stop times into frame indices (`frame0 = int(t_start*fps)`, etc.) and slice the movie accordingly.  
4. Cast to `float32` if needed to save memory.

### 3 Band-pass filtering  
* Design a causal or zero-phase band-pass filter (Butterworth or FIR).  
* Apply the filter **along the time axis only** (axis 0).  
* Keep the code readable: one helper that takes `(movie, fps, low, high)` and returns the band-pass movie.

### 4 Analytic signal & power  
* Compute the analytic signal per pixel via a Hilbert transform (`scipy.signal.hilbert` or equivalent).  
* Instantaneous power = absolute value squared of the analytic signal.  \

### 5 Visualisation  
Two figures only (displayed interactively; saving controlled by `--save`):

1. **Power movie animation**  
   * Use `matplotlib.animation` or `imageio` to play through time.  
   * Color map: `inferno` (or another perceptually uniform map).  
   * Scale intensity with a fixed `vmin/vmax` (e.g. 1st-99th percentile of the power volume) so brightness is comparable across frames.  
   * Overlay a “time = XX s” text in the corner for orientation.

2. **Peak-power map**  
   * Collapse the 3-D power array to a single 2-D image (e.g. maximum over time).  
   * Display with the same colour map and a colour-bar labelled “Band-power (a.u.)”.

If `--save` is **not** provided, just show the figures and exit cleanly.  
If `--save` *is* provided, write:
```text
<outdir>/power_movie.gif  or  .mp4
<outdir>/peak_power.png
```
Create `<outdir>` on the fly.

### 6 Debug-as-you-go guideline  
Claude should:
* Run short unit tests after each major section (loading, filtering, power) to verify shapes, dtypes, and numerical ranges.  
* Use `print()` or small assertions, then **remove any throw-away debug prints** before final commit.  
* Make sure the full script executes end-to-end on a small test movie without manual intervention.

### 7 Code style reminders  
* Keep the script **simple, clean, and interpretable** – favour clarity over maximal efficiency.  
* Add concise docstrings and inline comments where helpful.  
* Follow PEP-8 spacing / naming, but don’t over-engineer.  
* Avoid global variables outside a small “defaults” section.  
* Feel free to choose any standard scientific-Python libraries; no explicit dependency list needed.

---

## Acceptance checklist Claude should tick off

- [ ] CLI works with default arguments.
- [ ] Movie window extracts the correct time range.
- [ ] Band-pass filter behaves (inspect one pixel’s before/after spectrum during testing).
- [ ] Power movie displays without artefacts or frame rate mismatch.
- [ ] Peak-power map shows sensible spatial structure.
- [ ] Script exits gracefully when run with and without `--save`.
- [ ] Extraneous debug code removed; final file fits in ~150 lines.

When all items pass, save the finished script as `wave_detection/band_power_maps.py`.