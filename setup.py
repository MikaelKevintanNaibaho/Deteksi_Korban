import os
from setuptools import find_packages, setup

package_name = "krsri_video_driver"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # Add model and label files, explicitly listing them to avoid directories
        (os.path.join("share", package_name, "custom_model_lite"), [
            "krsri_video_driver/custom_model_lite/detect.tflite",
            "krsri_video_driver/custom_model_lite/labelmap.txt",
        ]),
        # Add the calibration file
        (os.path.join("share", package_name), ["krsri_video_driver/calibration_parameters.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="mikael",
    maintainer_email="mikael@todo.todo",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "image_publisher = krsri_video_driver.image_publisher:main",
            "image_subscriber = krsri_video_driver.image_subscriber:main",
            "raw_sub = krsri_video_driver.raw_sub:main",
            "coordinate_subs = krsri_video_driver.coordinate_subs:main",
        ],
    },
)
