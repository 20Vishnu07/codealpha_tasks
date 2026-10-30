# CodeAlpha_ObjectTracking
Real-time object detection and tracking (CodeAlpha AI internship, Task 4).

- Detection: YOLOv8 (pre-trained on COCO) through `ultralytics`
- Tracking: SORT implemented in `sort_tracker.py` (constant-velocity Kalman filter + Hungarian matching on IoU)
- Output: bounding boxes with class label and persistent tracking ID

```
pip install -r requirements.txt
python track.py                                   # webcam, press q to quit
python track.py --source video.mp4 --save out.mp4 # video file
python track.py --classes 0                       # people only
python test_sort.py                               # tracker unit test (no model needed)
```
Limitation: SORT uses motion only, so IDs can switch after long occlusions. Deep SORT adds appearance features to reduce that.
