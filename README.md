## Overview
This repository contains useful python scripts for training a YOLO segmentation model with Label Studio and SAM2.

### json_to_yolo.py
Convert Label Studio JSON brush labels to Ultralytics YOLO .txt files for segmentation model training.

Blog: https://medium.com/@siynnri/how-to-convert-label-studio-brush-labels-into-yolo-b4c64204787f


#### Install dependencies
```
pip install numpy opencv-python
```

#### Usage

1. Export annotations from Label Studio as a JSON file (e.g., `brush.json`).
2. Update `LABELS_MAPPING` with your class names and IDs.
3. Run the script:

```
python json_to_yolo.py
```

4. Find YOLO .txt files in the `labels/` directory.

Note: Labels that are neither brush nor polygon types are skipped, and their details are logged for review.