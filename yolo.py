import torch
import cv2
import numpy as np
from database import log_detection
from collections import defaultdict

class CentroidTracker:
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = defaultdict(int)
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])

        return self.objects


class YOLO:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.classes = self.model.names
        self.tracker = CentroidTracker()
        self.detected_boxes = []
        self.logged_objects = set()  # Set to keep track of logged object IDs

    def detect(self, frame):
        results = self.model(frame)
        detections = results.xyxy[0].numpy()

        self.detected_boxes = []

        rects = []
        for *xyxy, conf, cls in detections:
            class_index = int(cls)
            if class_index < 0 or class_index >= len(self.classes):
                print(f"Invalid class index detected: {class_index}")
                continue
            rects.append(xyxy)

        objects = self.tracker.update(rects)

        num_objects = 0

        for *xyxy, conf, cls in detections:
            startX, startY, endX, endY = xyxy
            class_index = int(cls)
            if class_index < 0 or class_index >= len(self.classes):
                continue

            centroid = ((startX + endX) / 2, (startY + endY) / 2)
            object_id = None
            for id, tracked_centroid in objects.items():
                if np.allclose(centroid, tracked_centroid, atol=1.0):
                    object_id = id
                    break

            if object_id is None:
                continue

            class_name = self.classes[class_index]
            num_objects += 1
            self.detected_boxes.append([startX, startY, endX, endY, object_id])
            label = f'{class_name} {conf:.2f} ID: {object_id}'
            frame = cv2.rectangle(frame, (int(startX), int(startY)), (int(endX), int(endY)), (255, 0, 0), 2)
            frame = cv2.putText(frame, label, (int(startX), int(startY) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Log detection only if it's a new object
            if object_id not in self.logged_objects:
                log_detection(class_name, conf, startX, startY, endX, endY, object_id)
                self.logged_objects.add(object_id)

        cv2.putText(frame, f'Counting: {num_objects}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        return frame

    def toggle_detection(self, cls_name, enable):
        if cls_name in self.detected_classes:
            self.detected_classes[cls_name] = enable
        else:
            print(f"Class {cls_name} not found in detected_classes")

    def get_detected_boxes(self):
        return self.detected_boxes
