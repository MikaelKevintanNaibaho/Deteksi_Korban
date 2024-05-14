from setuptools import find_packages, setup

package_name = "krsri_video_driver"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
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
