import numpy as np


class CoordinateTransformer:
    def __init__(self, focal_lenght, known_width, imW, imH):
        self.focal_lenght = focal_lenght
        self.known_width = known_width
        self.imW = imW
        self.imH = imH

    def calculate_distance(self, per_width):
        if per_width == 0:
            raise ValueError("Perceived width cannot be zero.")
        distance = (self.known_width * self.focal_lenght) / per_width
        return distance

    def pixel_to_camera_coordinate(self, bbox, distance, imW, imH):
        """
        Transform pixel coordinates to real-world coordinates in the camera coordinate system.
        
        Parameters:
        bbox (tuple): Bounding box coordinates (xmin, ymin, xmax, ymax).
        distance (float): Distance from the camera to the object.
        imW (int): Width of the image.
        imH (int): Height of the image.

        Returns:
        tuple: Real-world coordinates (x, y, z).
        """
        xmin, ymin, xmax, ymax = bbox
        image_center_x = self.imW / 2
        image_center_y = (ymin + ymax) / 2
        object_center_x = (xmin + xmax) / 2
        horizontal_angle = np.arctan(
            (object_center_x - image_center_x) / self.focal_lenght
        )
        vertical_angle = np.arctan(
            (image_center_y - self.focal_lenght) / self.focal_lenght
        )
        x = distance * np.sin(horizontal_angle)
        y = distance * np.sin(vertical_angle)
        z = distance

        return x, y, z
