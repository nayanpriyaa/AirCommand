# models/

Put your trained weights here as **`models/best.pt`** (the path in `config/gestures.yaml → model.path`).

Weights are not committed (see `.gitignore`); they come from `training/aircommand_training.ipynb`:

1. Run the notebook in Colab to the end. The last cell downloads `best.pt`
   (a copy is also saved to `MyDrive/AirCommand/best.pt` when Drive is enabled).
2. Move it here:
   - Windows PowerShell: `Move-Item $HOME\Downloads\best.pt .\models\best.pt`
   - Linux: `mv ~/Downloads/best.pt models/best.pt`
3. Start the app: `python -m src.main --dry-run`.

The app checks that the class list stored inside `best.pt` equals `model.classes` in the config
(same names, same order) and refuses to start otherwise. Generic `yolo11n.pt` / `yolov8n.pt`
COCO weights have no hand classes and are rejected by this check.

Use a different file with `python -m src.main --model path/to/weights.pt`.

Load the weights with the same Ultralytics minor version range used in Colab
(`requirements.txt`); a newer Ultralytics can usually read older weights, not always the reverse.
