import torch
import cv2
from database import log_detection

class YOLO:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.classes = self.model.names
        self.detected_boxes = []

    def detect(self, frame):
        results = self.model(frame)
        detections = results.xyxy[0].numpy()

        self.detected_boxes = []

        num_persons = 0

        for *xyxy, conf, cls in detections:
            class_index = int(cls)
            if class_index < 0 or class_index >= len(self.classes):
                print(f"Invalid class index detected: {class_index}")
                continue

            class_name = self.classes[class_index]

            # if class_name == 'person':
            #     num_persons += 1
            #     self.detected_boxes.append(xyxy)
            #     label = f'{class_name} {conf:.2f}'
            #     frame = cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
            #     frame = cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
            #     # Log detection
            #     log_detection(class_name, conf, xyxy[0], xyxy[1], xyxy[2], xyxy[3])
            
            num_persons += 1
            self.detected_boxes.append(xyxy)
            label = f'{class_name} {conf:.2f}'
            frame = cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
            frame = cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
            # Log detection
            log_detection(class_name, conf, xyxy[0], xyxy[1], xyxy[2], xyxy[3])

        cv2.putText(frame, f'Counting: {num_persons}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        return frame

    def toggle_detection(self, cls_name, enable):
        if cls_name in self.detected_classes:
            self.detected_classes[cls_name] = enable
        else:
            print(f"Class {cls_name} not found in detected_classes")

    def get_detected_boxes(self):
        return self.detected_boxes








# import torch
# import cv2
# from database import log_detection
# import numpy as np

# class YOLO:
#     def __init__(self):
#         self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
#         self.classes = self.model.names
#         self.detected_boxes = []
#         self.tracked_objects = {}  # Track object IDs and their positions
#         self.object_id = 0  # ID counter for new objects
#         self.line_position = 300  # Y-coordinate of the counting line
#         self.crossed_objects = set()  # Keep track of object IDs that crossed the line

#     def detect(self, frame):
#         results = self.model(frame)
#         detections = results.xyxy[0].numpy()

#         self.detected_boxes = []
#         current_objects = {}  # Current frame's objects

#         for *xyxy, conf, cls in detections:
#             class_index = int(cls)
#             if class_index < 0 or class_index >= len(self.classes):
#                 print(f"Invalid class index detected: {class_index}")
#                 continue

#             class_name = self.classes[class_index]
#             center_x = int((xyxy[0] + xyxy[2]) / 2)
#             center_y = int((xyxy[1] + xyxy[3]) / 2)

#             # if class_name == 'person':
#             #     current_objects[self.object_id] = (center_x, center_y)
#             #     self.object_id += 1
#             #     self.detected_boxes.append(xyxy)
#             #     label = f'{class_name} {conf:.2f}'
#             #     frame = cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
#             #     frame = cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#             #     log_detection(class_name, conf, xyxy[0], xyxy[1], xyxy[2], xyxy[3])


#             current_objects[self.object_id] = (center_x, center_y)
#             self.object_id += 1
#             self.detected_boxes.append(xyxy)
#             label = f'{class_name} {conf:.2f}'
#             frame = cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
#             frame = cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#             log_detection(class_name, conf, xyxy[0], xyxy[1], xyxy[2], xyxy[3])

#         # Draw the counting line
#         frame = cv2.line(frame, (0, self.line_position), (frame.shape[1], self.line_position), (0, 255, 0), 2)

#         # Update tracked objects
#         for object_id, (prev_x, prev_y) in self.tracked_objects.items():
#             if object_id in current_objects:
#                 curr_x, curr_y = current_objects[object_id]
#                 if prev_y < self.line_position <= curr_y or prev_y > self.line_position >= curr_y:  # Check if the object crossed the line
#                     if object_id not in self.crossed_objects:
#                         self.crossed_objects.add(object_id)

#         # Update tracked objects with current frame's objects
#         self.tracked_objects = current_objects.copy()

#         # Display the number of crossed objects
#         num_crossed = len(self.crossed_objects)
#         cv2.putText(frame, f'Count: {num_crossed}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

#         return frame

#     def toggle_detection(self, cls_name, enable):
#         if cls_name in self.detected_classes:
#             self.detected_classes[cls_name] = enable
#         else:
#             print(f"Class {cls_name} not found in detected_classes")

#     def get_detected_boxes(self):
#         return self.detected_boxes

