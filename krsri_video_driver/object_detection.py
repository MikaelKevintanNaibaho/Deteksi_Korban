import os
import cv2
import numpy as np
import importlib.util
import time
import yaml
from .coordinate_transformer import CoordinateTransformer


class ObjectDetection:
    def __init__(
        self,
        model_dir,
        calibration_file,
        graph="detect.tflite",
        labels="labelmap.txt",
        threshold=0.5,
        resolution="1280x720",
        use_TPU=False,
    ):
        self.MODEL_NAME = model_dir
        self.GRAPH_NAME = graph
        self.LABELMAP_NAME = labels
        self.min_conf_threshold = float(threshold)
        resW, resH = resolution.split("x")
        self.imW, self.imH = int(resW), int(resH)
        self.use_TPU = use_TPU
        self.calibration_file = calibration_file
        
        with open(self.calibration_file, 'r') as f:
            calibration_data = yaml.safe_load(f)
        
        camera_matrix = np.array(calibration_data['camera_matrix'])
        distortion_coefficients = np.array(calibration_data['distortion_coefficients'])

        # Initialize CoordinateTransformer with calibration parameters
        self.coordinate_transformer = CoordinateTransformer(
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion_coefficients,
            known_width=8.50,  # Assuming known width in centimeters
            imW=self.imW,
            imH=self.imH
        )

        # Import TensorFlow libraries
        pkg = importlib.util.find_spec("tflite_runtime")
        if pkg:
            from tflite_runtime.interpreter import Interpreter

            if self.use_TPU:
                from tflite_runtime.interpreter import load_delegate
        else:
            from tensorflow.lite.python.interpreter import Interpreter

            if self.use_TPU:
                from tensorflow.lite.python.interpreter import load_delegate

        # If using Edge TPU, assign filename for Edge TPU model
        if self.use_TPU:
            if self.GRAPH_NAME == "detect.tflite":
                self.GRAPH_NAME = "edgetpu.tflite"

        # Get path to current working directory
        self.CWD_PATH = os.getcwd()

        # Path to .tflite file and label map
        self.PATH_TO_CKPT = os.path.join(
            self.CWD_PATH, self.MODEL_NAME, self.GRAPH_NAME
        )
        self.PATH_TO_LABELS = os.path.join(
            self.CWD_PATH, self.MODEL_NAME, self.LABELMAP_NAME
        )

        # Load the label map
        with open(self.PATH_TO_LABELS, "r") as f:
            self.labels = [line.strip() for line in f.readlines()]

        # Load the Tensorflow Lite model
        if self.use_TPU:
            self.interpreter = Interpreter(
                model_path=self.PATH_TO_CKPT,
                experimental_delegates=[load_delegate("libedgetpu.so.1.0")],
            )
        else:
            self.interpreter = Interpreter(model_path=self.PATH_TO_CKPT)
        self.interpreter.allocate_tensors()

        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]["shape"][1]
        self.width = self.input_details[0]["shape"][2]

        self.floating_model = self.input_details[0]["dtype"] == np.float32

        self.input_mean = 127.5
        self.input_std = 127.5

        # Check output layer name to determine if this model was created with TF2 or TF1
        outname = self.output_details[0]["name"]

        if "StatefulPartitionedCall" in outname:  # This is a TF2 model
            self.boxes_idx, self.classes_idx, self.scores_idx = 1, 3, 0
        else:  # This is a TF1 model
            self.boxes_idx, self.classes_idx, self.scores_idx = 0, 1, 2

        self.start_time = time.time()
        self.frame_count = 0

    def perform_detection(self, frame):
        # Acquire frame and resize to expected shape [1xHxWx3]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.width, self.height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if self.floating_model:
            input_data = (np.float32(input_data) - self.input_mean) / self.input_std

        # Perform the actual detection by running the model with the image as input
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()

        # Retrieve detection results
        boxes = self.interpreter.get_tensor(
            self.output_details[self.boxes_idx]["index"]
        )[
            0
        ]  # Bounding box coordinates of detected objects
        classes = self.interpreter.get_tensor(
            self.output_details[self.classes_idx]["index"]
        )[
            0
        ]  # Class index of detected objects
        scores = self.interpreter.get_tensor(
            self.output_details[self.scores_idx]["index"]
        )[
            0
        ]  # Confidence of detected objects

        closest_distance = float("inf")  # Initialize with a very large value
        closest_x, closest_y, closest_z = 0, 0, 0  # Initialize with arbitrary values

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if (scores[i] > self.min_conf_threshold) and (scores[i] <= 1.0):
                # Get bounding box coordinates and draw box
                ymin = int(max(1, (boxes[i][0] * self.imH)))
                xmin = int(max(1, (boxes[i][1] * self.imW)))
                ymax = int(min(self.imH, (boxes[i][2] * self.imH)))
                xmax = int(min(self.imW, (boxes[i][3] * self.imW)))

                # Calculate center coordinates
                center_x = (xmin + xmax) // 2
                center_y = (ymin + ymax) // 2

                # Calculate width and height
                width = xmax - xmin
                height = ymax - ymin

                # Adjust bounding box coordinates based on image dimensions
                xmin = max(0, center_x - width // 2)
                ymin = max(0, center_y - height // 2)
                xmax = min(self.imW, center_x + width // 2)
                ymax = min(self.imH, center_y + height // 2)

                # Draw adjusted bounding box
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

                # Draw label
                object_name = self.labels[int(classes[i])]
                label = "%s: %d%%" % (object_name, int(scores[i] * 100))
                labelSize, baseLine = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(
                    frame,
                    (xmin, label_ymin - labelSize[1] - 10),
                    (xmin + labelSize[0], label_ymin + baseLine - 10),
                    (255, 255, 255),
                    cv2.FILLED,
                )
                cv2.putText(
                    frame,
                    label,
                    (xmin, label_ymin - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                )

                # If object is "korban", perform additional calculations
                if object_name == "korban":
                    # Calculate distance and coordinates
                    distance = self.coordinate_transformer.calculate_distance(width)
                    closest_distance = min(closest_distance, distance)
                    (
                        closest_x,
                        closest_y,
                        closest_z,
                    ) = closest_x, closest_y, closest_z = self.coordinate_transformer.pixel_to_camera_coordinate(
    (xmin, ymin, xmax, ymax), distance
)

                    cv2.putText(
                        frame,
                        "Distance: {:.2f} cm".format(distance),
                        (xmax - labelSize[0], ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                    )

        # Draw additional information
        coordinate_text = "X: {:.2f}, Y: {:.2f}, Z: {:.2f}".format(
            closest_x, closest_y, closest_z
        )
        text_width = cv2.getTextSize(coordinate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[
            0
        ][0]
        text_x = self.imW - text_width - 10
        cv2.putText(
            frame,
            coordinate_text,
            (text_x, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

        return frame, closest_x, closest_y, closest_z
