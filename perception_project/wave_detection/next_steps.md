# Update v3 - Analytic-signal cube & phase-gradient analysis  
*Script:* `wave_detection/band_power_maps.py`  
*Notebook:* `wave_detection/band_power_analysis.ipynb`  
(You will **only** write functions inside `band_power_maps.py`; the notebook is for visualisation calls.)

---

## 1 - Scope of this revision
1. **Keep** all existing functionality (movie load → band-pass → Hilbert power maps → side-by-side animation with stimulus cue).  
2. **Add**:
   * Construction of the **full analytic-signal cube** for each band (complex data, A e^{iφ}).  
   * Computation of a **phase-gradient vector field** per frame and associated speed/direction metrics.  
3. **Expose** lightweight helper calls that the notebook can import to:
   * Plot a single-frame quiver overlay on the movie.  
   * Generate a polar histogram of propagation directions over a time window.  

---

## 2 - Methodological steps Claude must implement

### 2.1  Analytic-signal cube
1. **Input**: band-pass movie segment (T, Y, X) and frame-rate `fps`.  
2. **Operation**: apply a Hilbert transform along the **time axis only** to obtain the complex analytic signal **Z**.  
3. **Outputs** stored in a dict (or dataclass) returned to the caller:  
   * `Z` – complex cube  
   * `amplitude` = |Z|  
   * `phase`      = angle(Z)  
   * `inst_freq`  = frame-to-frame phase derivative / (2πΔt)  (useful for QC)

### 2.2  Phase-gradient vector field (per frame)
For each frame *t*:
1. Extract 2-D phase map φ(x,y).  **Optional**: spatial Gaussian blur (σ ≈ 1 px) *before* gradients to suppress pixel noise.  
2. Compute spatial gradients: ∂φ/∂x and ∂φ/∂y (central differences or `np.gradient`).  
3. **Direction unit-vector** d = −∇φ / ‖∇φ‖ (sign puts arrows in travel direction).  
4. **Speed** v = (∂φ/∂t) / (2π ‖∇φ‖) where ∂φ/∂t is phase difference to the next frame divided by Δt.  
5. Package results as arrays with shape (Y, X): `grad_x`, `grad_y`, `speed`, `angle`.

### 2.3  Minimal QC metrics (no plotting in the script)
Return alongside the vector field:
* Mean and median speed per frame.  
* Circular mean direction per frame (use `scipy.stats.circmean` or equivalent).  
These are cheap sanity checks for the notebook.

---

## 3 - API layout expected inside **band_power_maps.py**

| Function | Purpose | Returned objects |
|----------|---------|------------------|
| `build_analytic_cube(movie, fps)` | Wraps §2.1 | dict with keys: `Z`, `amplitude`, `phase`, `inst_freq` |
| `phase_gradient_cube(phase, fps, blur_sigma=1)` | Implements §2.2 for all frames | dict with keys: `grad_x`, `grad_y`, `speed`, `angle`, `mean_speed`, `mean_dir` |
| `pipeline_analyse_band(movie, fps, band)` | One-stop call used by notebook:<br>band-pass → analytic cube → gradient cube | returns a `results` dict ready for visualisation |

*(Re-export any existing helper names so old notebook cells do not break.)*

---

## 4 - Notebook expectations (`band_power_analysis.ipynb`)

Claude **does not** code the notebook here, but must ensure functions let the notebook author:

1. **Visual check**  
   ```python
   res = pipeline_analyse_band(movie, fps, band=(4,8))
   show_quiver_on_frame(res['amplitude'][123], res['grad_x'][123], res['grad_y'][123])
   ```
   where `show_quiver_on_frame` is a tiny plotting helper inside the script.

2. **Direction statistics**  
   ```python
   directions = res['angle'].reshape(-1)  # flatten over space & time
   plot_direction_histogram(directions)
   ```

3. **Overlay speed trace** on existing stimulus timeline plot.

Claude may provide the short plotting helpers **inside the script** so the notebook can just import and call them.

---

## 5 - Algorithmic & numerical guidance

* **Phase unwrapping**: do **not** unwrap in time; gradients work on wrapped phase.  If unwrap in space is easier, Claude may use `np.unwrap` along both spatial axes *before* gradients.  
* **Gradient edge handling**: simplest is second-order central diff with forward/backward at borders.  
* **Speed units**: final `speed` array should be in *pixels per second*.  If pixel size in μm becomes available later, conversion lives in the notebook, not the script.  
* **Computational cost**: keep loops in NumPy vectorised form; no Python loops over pixels.  

---

## 6 - Testing & debugging protocol for Claude

1. **Unit test** analytic cube: feed a synthetic sine wave movie; verify amplitude is constant ≈1, phase advances linearly, inst_freq ≈ input freq.  
2. **Visual test** gradient: create a synthetic plane wave; direction histogram should peak at the implanted direction; mean speed should match ground truth.   
3. Remove all print-spam / debug plots before final commit.

---

## 7 - Deliverables

* **band_power_maps.py** updated with new functions, docstrings, and the minimal QC helpers.   
* No embedded notebook cells or top-level test code in the script.  
* Internal comments where algorithms might be opaque (phase wrap logic, gradient maths).

When the above passes the tests, Claude can declare the implementation complete.