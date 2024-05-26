import numpy as np
import cv2

class CoordinateTransformer:
    def __init__(self, camera_matrix, distortion_coefficients, known_width, imW, imH):
        self.camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients
        self.known_width = known_width
        self.imW = imW
        self.imH = imH


    def calculate_distance(self, per_width):
        """
        Calculate the distance to an object based on its perceived width in the image.

        Parameters:
        per_width (float): Perceived width of the object in pixels.

        Returns:
        float: Distance to the object in the same units as the known width.
        """
        if per_width == 0:
            raise ValueError("Perceived width cannot be zero.")
        # Use calibrated focal length for distance calculation
        distance = (self.known_width * self.camera_matrix[0][0]) / per_width
        return distance


    def pixel_to_camera_coordinate(self, bbox, distance):
        """
        Transform pixel coordinates to real-world coordinates in the camera coordinate system.
        
        Parameters:
        bbox (tuple): Bounding box coordinates (xmin, ymin, xmax, ymax).
        distance (float): Distance from the camera to the object.

        Returns:
        tuple: Real-world coordinates (x, y, z).
        """
        xmin, ymin, xmax, ymax = bbox

        # Adjust for distortion
        undistorted_points = cv2.undistortPoints(
            np.array([[(xmin + xmax) / 2, (ymin + ymax) / 2]], dtype=np.float32),
            self.camera_matrix,
            self.distortion_coefficients,
        )

        # Project to 3D
        object_center = np.array([undistorted_points[0][0][0], undistorted_points[0][0][1], 1])
        world_coords = np.dot(np.linalg.inv(self.camera_matrix), object_center) * distance

        x, y, z = world_coords

        return x, y, z
