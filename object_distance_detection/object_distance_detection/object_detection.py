import numpy as np
from ultralytics import YOLO
import random
import colorsys
import torch
import cv2


class ObjectDetection:
    def __init__(self, weights_path=r"D:\\WorkSpace\\robocon\\train_model\\runs\\detect\\train\\weights\\best.pt"):
        self.weights_path = weights_path

        print("Torch version:", torch.__version__)
        print("Is CUDA available?", torch.cuda.is_available())

        # Load Yolo model
        self.model = YOLO(self.weights_path)

        # Load Default device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")  # Use "cuda" for GPU
        else:
            self.device = torch.device("cpu")

        # Move model to the correct device
        self.model.to(self.device)

        # Get the class names from the model
        self.classes = self.model.names

        # Random colors for each class
        self.colors = self.random_colors(len(self.classes))

    def random_colors(self, N, bright=False):
        """Generate random colors."""
        brightness = 255 if bright else 180
        hsv = [(i / N + 1, 1, brightness) for i in range(N + 1)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors

    def detect(self, frame, imgsz=1280, conf=0.25, nms=True, classes=None):
        """Perform object detection on a frame."""
        # Resize frame to 640x640
        frame_resized = cv2.resize(frame, (640,640))

        # Convert the resized frame to a tensor with shape (1, 3, 640, 640)
        frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # Move the tensor to the appropriate device
        frame_tensor = frame_tensor.to(self.device)

        # Filter classes if specified
        filter_classes = classes if classes else None

        # Perform detection
        results = self.model.predict(source=frame_tensor, save=False, save_txt=False,
                                     imgsz=imgsz, conf=conf, nms=nms, classes=filter_classes,
                                     half=False, device=self.device)

        # Get the first result from the array (we are only using one image)
        result = results[0]

        # Get bounding boxes, class ids, and confidence scores
        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
        scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)

        return bboxes, class_ids, scores

    def draw_object_info(self, color_image):
        """Draw bounding boxes and class names on the image."""
        # Get the object detection
        bboxes, class_ids, scores = self.detect(color_image)

        for bbox, class_id, score in zip(bboxes, class_ids, scores):
            x, y, x2, y2 = bbox
            color = (int(self.colors[class_id][0] * 255), int(self.colors[class_id][1] * 255), int(self.colors[class_id][2] * 255))

            # Draw rectangle around the object
            cv2.rectangle(color_image, (x, y), (x2, y2), color, 2)

            # Display class name and confidence score
            class_name = self.classes[class_id]
            cv2.putText(color_image, f"{class_name} {score:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        return color_image
