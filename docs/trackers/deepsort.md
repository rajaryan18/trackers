---
comments: true
---

# DeepSORT

## Overview

DeepSORT (Simple Online and Realtime Tracking with a Deep Association Metric) extends the SORT framework by incorporating appearance information. It uses a deep learning model to extract features (embeddings) from bounding boxes, which are then used to match detections to tracks over time. This significantly reduces identity switches and improves tracking robustness in the presence of occlusions and complex motion.

DeepSORT employs a Kalman filter for motion prediction and a matching cascade algorithm that prioritizes tracks with more recent updates. It also uses a Mahalanobis distance metric to gate the association process, ensuring that detections are only matched to tracks that are motion-compatible.

## Comparison

For comparisons with other trackers, plus dataset context and evaluation details, see the [tracker comparison](comparison.md) page.

|  Dataset  | HOTA | IDF1 | MOTA |
| :-------: | :--: | :--: | :--: |
|   MOT17   | 42.7 | 43.8 | 41.5 |

## Run on video, webcam, or RTSP stream

These examples use `opencv-python` for decoding and display. Replace `<SOURCE_VIDEO_PATH>`, `<WEBCAM_INDEX>`, and `<RTSP_STREAM_URL>` with your inputs. `<WEBCAM_INDEX>` is usually 0 for the default camera.

=== "Video"

    ```python
    import cv2
    import supervision as sv
    from rfdetr import RFDETRMedium
    from trackers import DeepSORTTracker

    tracker = DeepSORTTracker()
    model = RFDETRMedium()

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    video_capture = cv2.VideoCapture("<SOURCE_VIDEO_PATH>")
    if not video_capture.isOpened():
        raise RuntimeError("Failed to open video source")

    while True:
        success, frame_bgr = video_capture.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = model.predict(frame_rgb)
        
        # Note: DeepSORT performs better when features are provided.
        # If your model doesn't provide features, DeepSORT falls back to 
        # motion-based tracking (Mahalanobis distance).
        detections = tracker.update(detections)

        annotated_frame = box_annotator.annotate(frame_bgr, detections)
        annotated_frame = label_annotator.annotate(
            annotated_frame,
            detections,
            labels=[f"#{id}" for id in detections.tracker_id],
        )

        cv2.imshow("RF-DETR + DeepSORT", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    ```

=== "Webcam"

    ```python
    import cv2
    import supervision as sv
    from rfdetr import RFDETRMedium
    from trackers import DeepSORTTracker

    tracker = DeepSORTTracker()
    model = RFDETRMedium()

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    video_capture = cv2.VideoCapture("<WEBCAM_INDEX>")
    if not video_capture.isOpened():
        raise RuntimeError("Failed to open webcam")

    while True:
        success, frame_bgr = video_capture.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = model.predict(frame_rgb)
        detections = tracker.update(detections)

        annotated_frame = box_annotator.annotate(frame_bgr, detections)
        annotated_frame = label_annotator.annotate(
            annotated_frame,
            detections,
            labels=[f"#{id}" for id in detections.tracker_id],
        )

        cv2.imshow("RF-DETR + DeepSORT", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    ```

=== "RTSP"

    ```python
    import cv2
    import supervision as sv
    from rfdetr import RFDETRMedium
    from trackers import DeepSORTTracker

    tracker = DeepSORTTracker()
    model = RFDETRMedium()

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    video_capture = cv2.VideoCapture("<RTSP_STREAM_URL>")
    if not video_capture.isOpened():
        raise RuntimeError("Failed to open RTSP stream")

    while True:
        success, frame_bgr = video_capture.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = model.predict(frame_rgb)
        detections = tracker.update(detections)

        annotated_frame = box_annotator.annotate(frame_bgr, detections)
        annotated_frame = label_annotator.annotate(
            annotated_frame,
            detections,
            labels=[f"#{id}" for id in detections.tracker_id],
        )

        cv2.imshow("RF-DETR + DeepSORT", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    ```
