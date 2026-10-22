# AirCommand

Control your desktop with hand gestures from a webcam. A YOLO11n detector fine-tuned on
HaGRIDv2 recognises five static gestures; a deterministic, unit-tested state machine turns
them into cursor movement, clicks, and slide navigation behind a latched safety layer.

```
webcam ─► Camera ─► GestureDetector (YOLO) ─► HandSelector ─► GestureStabilizer ─► GestureEngine ─► ActionExecutor ─► PyAutoGUI
         camera.py   detector.py             gesture_logic.py (pure, no I/O)                         cursor_controller.py
                                                                                   ▲
                                              SafetyController + GlobalHotkeys ────┘  safety.py (gates every action)
```

---

## 1. Design decisions (read this first)

| Decision | Why |
|---|---|
| **Click = point, then make a fist and hold ~0.25 s** (not pinch) | Pinch needs fingertip distance. YOLO outputs boxes, not finger joints, so a "pinch" from boxes would be guesswork. The fist-after-pointing click uses only what the detector actually provides and is gated by majority voting, a hold timer, an edge trigger, a cooldown, and a "must have been pointing within 1.2 s" context check. |
| **No MediaPipe** | Adding landmarks would make pinch possible but doubles the inference stack and the failure surface. MediaPipe is the right tool if you later want pinch/scroll; it is not required for this vocabulary. |
| **Cursor position = estimated fingertip from the `one` box** (top-centre, `cursor.anchor_y`) | Honest limitation: this is a heuristic, not a landmark. One Euro filtering and a dead zone make it usable; precision is lower than a mouse. |
| **Swipe = fast horizontal motion of the `palm` box** | Static-image datasets have no swipe class. Swipe is computed from box-centre motion over 0.55 s (distance, straightness, monotonicity) so a static palm never triggers it. |
| **HaGRIDv2, 6 classes** (`one, fist, palm, like, peace, no_gesture`) | Real, public, subject-split dataset with bounding boxes and a `no_gesture` class that cuts false positives. Chosen gestures are visually distinct (no `palm`/`stop` look-alikes). |
| **Global hotkeys via pynput** | `cv2.waitKey` only sees keys while the preview window is focused, which it loses the moment you click another app. An Esc that only works in one window is not an emergency stop. |
| **Starts INACTIVE, e-stop is latched** | Arming is always explicit. After e-stop you must clear it (F9 / `r`) and then arm again (F8 / `a`). |

Structure additions beyond the requested layout: `src/overlay.py` (drawing kept out of
`main.py`), `tests/test_config.py`, `tests/test_cursor_controller.py`,
`tests/test_pipeline.py`, `tests/test_dataset_setup.py`, `tests/test_detector.py`, `tests/_helpers.py`, `pytest.ini`,
`requirements-dev.txt`, and empty `__init__.py` files so `src`, `training` and `tests` import as packages.

```
AirCommand/
├── README.md
├── requirements.txt            requirements-dev.txt   pytest.ini   .gitignore
├── config/gestures.yaml        every tunable; validated at startup (unknown keys rejected)
├── training/
│   ├── aircommand_training.ipynb   Colab: dataset → fine-tune → val/test eval → export
│   └── dataset_setup.py            HaGRIDv2 → YOLO format, subject-disjoint, partial download
├── src/
│   ├── main.py               CLI, main loop, AirCommandApp pipeline, benchmark/session stats
│   ├── camera.py             webcam/video capture, backend fallback, latest-frame thread
│   ├── detector.py           the only Ultralytics import (lazy); Detection dataclass
│   ├── gesture_logic.py      hand selection, voting, hold triggers, swipe, action emission
│   ├── cursor_controller.py  One Euro filter, screen mapping, PyAutoGUI/Null backends, executor
│   ├── safety.py             arm/disarm/e-stop state machine, rate limit, global hotkeys
│   ├── overlay.py            on-screen UI
│   └── config.py             typed config + validation
├── tests/                    unit tests; no model, camera or display needed
├── models/README.md          where best.pt goes
├── docs/                     put result images / demo GIF here
└── outputs/.gitkeep          session/benchmark JSON, screenshots, recordings
```

## 2. Gesture → action mapping

| Gesture (HaGRID class) | Looks like | Action | Guard |
|---|---|---|---|
| `one` | index finger up | move cursor (absolute, inside the cyan active region) | moves only when the current frame **and** the 3-of-5 vote agree |
| `one` → `fist` | point, then close hand | **left click** (once) | vote + 0.25 s hold + pointing within 1.2 s + release required + 0.7 s cooldown |
| `peace` | V sign | **right click** (once) | vote + 0.45 s hold + release + cooldown |
| `palm` + fast move left | open palm sweep | **Right arrow** (next slide) | ≥22 % of frame width in ≤0.55 s, mostly horizontal, monotonic; 1 s cooldown |
| `palm` + fast move right | open palm sweep | **Left arrow** (previous slide) | same |
| `like` held 1.5 s | thumbs up | **disarm** (arm too only if `safety.allow_gesture_arming: true`) | edge-triggered |
| `no_gesture` | relaxed / other hand | nothing | trained explicitly to absorb false positives |

Gestures can only press `left, right, up, down, pageup, pagedown` (`config.ALLOWED_KEYS`,
enforced at config load **and** at execution). There is no typing, hotkey chording or shell access.

## 3. Safety and keyboard shortcuts

| Key | Scope | Effect |
|---|---|---|
| **Esc** | global (+ window) | **EMERGENCY STOP** — latched, blocks every action |
| **F9** / `r` | global / window | clear e-stop → INACTIVE (does **not** arm) |
| **F8** / `a` | global / window | arm (F8 toggles arm/disarm) |
| `d` | window | disarm |
| `q` | window | **quit** (also Ctrl+C in the terminal, or close the window) |
| `h` / `s` | window | toggle help / save screenshot to `outputs/` |
| mouse → any screen corner | system | PyAutoGUI fail-safe → e-stop |

Additional guards: 0.75 s grace after arming; gestures held while arming must be released
first; ≤3 clicks/keys per second hard cap; auto-disarm after 120 s without a hand; any
unexpected desktop-backend error triggers e-stop (fail closed); the app cursor is clamped
8 px from screen edges so it never trips the fail-safe itself. The global listener does not
swallow keys: Esc also reaches the focused app (it will exit a PowerPoint slideshow).

Banner colours: grey = INACTIVE, green = ACTIVE, red = E-STOP. `[DRY-RUN]` = nothing touches the desktop.

## 4. Prerequisites

- Python **3.10–3.12** (64-bit). 3.11 is the safest choice for PyTorch wheels.
- A webcam (720p is plenty; 640×480 is used by default).
- Windows 10/11, or Linux with an **X11** session (see limitations for Wayland). macOS is untested.
- A Google account for Colab training (free T4 GPU tier works).

## 5. Installation

**Windows PowerShell**
```powershell
cd AirCommand
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

**Linux (Ubuntu/Debian)**
```bash
sudo apt install python3-venv python3-tk python3-dev   # tk/dev are needed by PyAutoGUI
cd AirCommand
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

`pip install ultralytics` pulls a CPU PyTorch build on Windows. For an NVIDIA GPU, first
install the CUDA build with the command from https://pytorch.org/get-started/locally/, then
install the requirements. CPU inference of YOLO11n at 640 is usable on most recent laptops;
measure yours with the benchmark command below.

Verify the install (no webcam or weights needed):
```bash
python -m pytest
```

## 6. Dataset: HaGRIDv2

- Source: https://github.com/hukenovs/hagrid (HaGRIDv2, 1.08 M FullHD images, 33 gestures + `no_gesture`).
- License: a variant of **CC BY-SA 4.0** — read `license/` in that repo before redistributing data or derived weights.
- Annotations: per-class JSON, key = image id, normalized COCO-style `[x, y, w, h]` boxes, `labels`, `user_id`.
- Official train/val/test split is **by subject (`user_id`)**, which prevents identity leakage.

`training/dataset_setup.py`:
1. Reads the central directory of the 119 GB `hagridv2_512.zip` and of `annotations.zip` using HTTP Range requests.
2. Samples (seeded) 2500 / 400 / 600 images per class from the official train / val / test splits.
3. Drops any image containing a hand label outside the vocabulary (keeping it would teach "hand = background").
4. Verifies no `user_id` or image id occurs in two splits and removes violators (expected count: 0).
5. Downloads only the selected images (≈1–2 GB; resumable, CRC-checked), writes YOLO labels, `data.yaml` and `split_report.json`.

The notebook runs this for you. To run it locally instead:
```bash
python training/dataset_setup.py --out datasets/aircommand                     # remote, default quotas
python training/dataset_setup.py --list-only                                   # inspect archive layout
python training/dataset_setup.py --images-zip hagridv2_512.zip --annotations-zip annotations.zip   # local archives
python training/dataset_setup.py --images-dir hagrid/images --annotations-dir hagrid/annotations    # extracted
```
No account is needed. If the HaGRID host stops honouring Range requests, the script stops
with that message; download the archives manually from the links in the HaGRID README and use
the local options above (needs ~120 GB free disk for the 512 px archive).

## 7. Training in Google Colab

1. Open https://colab.research.google.com → *Upload* → `training/aircommand_training.ipynb`.
2. *Runtime → Change runtime type → T4 GPU*.
3. Zip this repository as `AirCommand.zip` (or push it to GitHub and set `REPO_URL` in cell 1).
4. *Runtime → Run all*. Upload the ZIP when prompted; allow Google Drive access (checkpoints,
   dataset cache, and `best.pt` are saved to `MyDrive/AirCommand/` so a disconnect is not fatal).
5. If the session dies mid-training: re-run the settings and dataset cells, then the **resume** cell.

Defaults: `yolo11n.pt` (COCO-pretrained), 40 epochs, imgsz 640, batch 32, patience 15, cosine LR.
Set `MODEL_NAME = "yolov8n.pt"` to train YOLOv8n instead. Training time depends on the GPU
Colab assigns and on the download speed from the HaGRID host; it has not been measured here.

Evaluation produced by the notebook (all from your run):
- val split: per-class precision, recall, mAP50, mAP50-95 (used for `best.pt` selection only)
- **test split** (held-out subjects): the same table, normalized confusion matrix, PR curve, top confusions
- inference speed: Ultralytics GPU timing + a 100-image CPU `predict()` benchmark
- `aircommand_metrics.json` summarising everything

## 8. Install the trained weights

The notebook's last cell downloads `best.pt` and `aircommand_metrics.json`.

```powershell
Move-Item $HOME\Downloads\best.pt .\models\best.pt        # Windows
```
```bash
mv ~/Downloads/best.pt models/best.pt                      # Linux
```
The app verifies the class list inside the weights matches `model.classes` (order included).

## 9. Run

Always start with a dry run: identical pipeline, actions printed instead of executed.
```bash
python -m src.main --dry-run
```
Live control (starts INACTIVE; press **F8** or `a` to arm, **Esc** to stop):
```bash
python -m src.main
```
Useful options (same on PowerShell and Linux):
```bash
python -m src.main --camera 1                    # another webcam
python -m src.main --model models/yolov8n_best.pt
python -m src.main --conf 0.6 --sensitivity 1.3
python -m src.main --source demo.mp4 --dry-run   # recorded video instead of webcam
python -m src.main --dry-run --record outputs/demo.mp4   # record the annotated preview for a demo
python -m src.main --benchmark 300 --dry-run     # measure FPS/latency → outputs/benchmark_*.json
python -m src.main --help
```
Every run writes `outputs/session_*.json` (FPS, latency p50/p95, action counts, package versions).

Recommended first session: dry run → watch the banner, boxes and progress bars → point with
`one` and watch the cursor dot → point then fist (click bar fills, "LEFT CLICK" appears) →
palm swipe in front of the camera → thumbs-up 1.5 s. Then open a slideshow, run live, arm, swipe.

## 10. Tuning (config/gestures.yaml)

| Problem | Change |
|---|---|
| Cursor jittery at rest | lower `cursor.min_cutoff` (e.g. 0.8) or raise `cursor.deadzone_px` |
| Cursor lags behind fast moves | raise `cursor.beta` (e.g. 0.02) |
| Too much hand travel needed | raise `cursor.sensitivity` (1.3–1.6) or enlarge `cursor.active_region` |
| Accidental clicks | raise `model.per_class_conf.fist`, `actions.click_hold_s` |
| Clicks feel slow | lower `actions.click_hold_s` to 0.18 (keep `smoothing.min_votes` ≥ 3) |
| Swipes missed | lower `swipe.min_dx` to 0.18 or `swipe.min_points` to 3; swipe nearer the camera |
| Swipes fire on their own | raise `swipe.min_dx`, `swipe.monotonic_ratio` |
| Want thumbs-up to arm too | `safety.allow_gesture_arming: true` (off by default on purpose) |
| Swap slide direction | exchange `swipe.left_key` / `swipe.right_key` |

## 11. Testing and evaluation

```bash
python -m pytest                       # all tests (pytest)
python -m unittest discover -s tests -t .   # same tests without pytest
```
Covered: majority-vote smoothing, single-frame spikes, click hold/edge/cooldown/context,
right click, swipe direction / slow drift / vertical motion / cooldown, toggle, reset-on-arm,
safety state machine, latched e-stop, cross-thread e-stop, rate limit, idle disarm, key
allow-list, fail-closed backend errors, PyAutoGUI fail-safe, One Euro filter, screen mapping and
clamping, config validation, detector box parsing and missing-weights errors, a full pipeline run
with a scripted fake detector, and HaGRID → YOLO conversion including leakage removal and partial ZIP reads.

Model quality: notebook sections 5–6. Live system speed: `--benchmark`. For a gesture-level
check, record a short clip per gesture (`--record`) and replay it with `--source clip.mp4 --dry-run`
to count triggered vs. intended actions.

## 12. Troubleshooting

**Webcam**
- `Could not open camera index 0` → close Zoom/Teams/OBS/browser tabs using it; try `--camera 1`.
- Windows: *Settings → Privacy & security → Camera* → allow desktop apps. Try `camera.backend: msmf` or `dshow`.
- Linux: `ls /dev/video*`; `sudo usermod -aG video $USER` then re-login; try `camera.backend: any`.
- Black/slow first frames on Windows → `dshow` is tried first by `auto`; set `msmf` if your camera prefers it.

**Dependencies**
- `cv2.imshow ... not implemented` → `opencv-python-headless` is installed alongside `opencv-python`:
  `pip uninstall -y opencv-python-headless opencv-python && pip install opencv-python`.
- PyTorch DLL errors on Windows → install the Microsoft Visual C++ Redistributable (x64) and reinstall torch.
- `KeyError: 'DISPLAY'` / Xlib errors (Linux) → run from a desktop X11 session, not SSH; install `python3-tk python3-dev`.

**Model weights**
- `Model weights not found` → put `best.pt` in `models/` or pass `--model`.
- `Class list in the weights does not match` → you are loading COCO weights or a model trained
  with different classes; fix `model.classes` order or retrain.
- `Failed to load ... best.pt` → re-download (partial file) and keep Ultralytics within `requirements.txt`.

**Permissions / desktop control**
- Clicks do nothing in some windows on Windows → apps running **as Administrator** ignore input from
  non-elevated processes; run AirCommand elevated too, or use non-elevated targets.
- Cursor lands at the wrong place on high-DPI Windows → set display scaling to 100 % to confirm,
  then report your scaling; PyAutoGUI marks the process DPI-aware, but mixed-DPI multi-monitor setups are not handled.
- Multiple monitors → mapping covers the primary screen only.
- `Global hotkeys unavailable` → pynput could not start (Wayland, no X server). Esc then works
  only while the AirCommand window is focused; the mouse-to-corner fail-safe still works on X11/Windows.

**Dataset script**
- `did not honour an HTTP Range request` → use the manual archive options in section 6.
- `No images for classes [...]` → the archive layout changed; run `--list-only` and adjust.
- Failed downloads → re-run the same command; finished files are cached.

## 13. Known limitations

- Cursor precision is limited by box-based fingertip estimation; fine targets (small buttons) are hard.
- Fast swipes cause motion blur; detections can drop out. The detector uses raw frames and tolerates gaps, but very fast swipes may be missed.
- One active hand. A second gesturing hand can take over tracking if the first disappears.
- Lighting and distance outside HaGRID's distribution (very dark rooms, > ~3 m) degrade accuracy.
- Linux **Wayland**: PyAutoGUI and pynput only see/act on XWayland apps; use an X11 session ("Ubuntu on Xorg").
- macOS: not tested; needs Accessibility and Input Monitoring permissions for PyAutoGUI/pynput.
- Ultralytics is **AGPL-3.0**. If you distribute this project or a service built on it, AGPL obligations apply (or obtain an Ultralytics enterprise license).
- Verification status: executed during development — all unit tests; the dataset converter (synthetic HaGRID-shaped archives, ZIP64 headers, partial reads through a local HTTP Range server); the full `src.main` loop with a scripted fake detector on a video file, headless and windowed, with real PyAutoGUI 0.9.54 and pynput 1.8 hotkeys on a virtual X11 display (Xvfb). **Not** executed: Ultralytics/YOLO inference, the Colab notebook, the remote HaGRID download, a physical webcam, Windows. Those need your hardware and accounts.

## 14. Results (fill in with your own measurements)

> Replace every `TBD` with numbers from `aircommand_metrics.json` and `outputs/benchmark_*.json`. Do not copy numbers from elsewhere.

**Test split (held-out subjects)** — model: `TBD`, epochs trained: `TBD`, imgsz 640

| class | precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| one | TBD | TBD | TBD | TBD |
| fist | TBD | TBD | TBD | TBD |
| palm | TBD | TBD | TBD | TBD |
| like | TBD | TBD | TBD | TBD |
| peace | TBD | TBD | TBD | TBD |
| no_gesture | TBD | TBD | TBD | TBD |
| **all** | TBD | TBD | TBD | TBD |

Confusion matrix: `docs/confusion_matrix_test.png` (copy from `runs/.../eval_test/`).

**Live performance** (`python -m src.main --benchmark 300 --dry-run`)

| machine | device | FPS | inference p50 / p95 (ms) |
|---|---|---|---|
| TBD | TBD | TBD | TBD |

**Demo**: `docs/demo.gif` / screenshots from `s` key in `outputs/`.

## 15. End-to-end smoke test

1. `python -m pytest` → all tests pass.
2. Colab: run all cells → `split_report.json` shows `user_overlap_before` all `0` and non-zero counts for every class; test table printed; `best.pt` downloaded.
3. `models/best.pt` in place → `python -m src.main --dry-run` → window opens, grey **INACTIVE** banner, boxes follow your hand.
4. Press `a` → green **ACTIVE [DRY-RUN]**; point (`one`) → cyan region + yellow dot; point → fist → terminal prints `[dry-run] left click` once.
5. Palm swipe left → `[dry-run] key right`; hold thumbs-up 1.5 s → banner grey (disarmed).
6. Press Esc → red **E-STOP**; `a` is refused until `r`.
7. Live: `python -m src.main`, click into another app, press **F8** (arms), then **Esc** from that app → e-stop works without focusing the preview window.
8. `python -m src.main --benchmark 300 --dry-run` → `outputs/benchmark_*.json` written; copy numbers into section 14.

## Credits

- HaGRID / HaGRIDv2: Kapitanov et al., WACV 2024; Nuzhdin et al., arXiv:2412.01508.
- Ultralytics YOLO (AGPL-3.0). One Euro filter: Casiez, Roussel, Vogel, CHI 2012.
