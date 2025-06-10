import os
import glob
from setuptools import setup, find_packages


package_name = "grip"
share_dir = os.path.join("share", package_name)

### Crawl files
setup_py_dir = os.path.dirname(os.path.realpath(__file__))

need_files = []
script_files = []
datadir = "grip_assets"

hh = setup_py_dir + "/" + datadir
hh2 = setup_py_dir + "/examples"

for root, dirs, files in os.walk(hh):
    for fn in files:
        ext = os.path.splitext(fn)[1][1:]
        if (
            ext
            and ext
            in "yaml index meta data-00000-of-00001 png gif jpg urdf sdf obj txt mtl dae off stl STL xml cfg pt pth".split()
        ):
            fn = root + "/" + fn
            need_files.append(fn[1 + len(hh) :])


setup(
    name=f"{package_name}x",
    version="0.0.13",
    description="Grip is a prototyping toolbox for manipulation research.",
    long_description=open("README.md").read(),
    url="https://github.com/eaa3/grip.git",
    author="Ermano Arruda",
    maintainer="Ermano Arruda",
    maintainer_email="ermano.arruda@gmail.com",
    license="MIT",
    python_requires=">=3.10,<3.13",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
    ],
    tests_require=["pytest"],
    packages=find_packages(exclude=["test", "test.robot", "test.sensors"]),
    package_dir={"": "."},
    package_data={"grip_assets": need_files},
    data_files=[
        (share_dir, ["package.xml"]),
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        (share_dir + "/launch", glob.glob(os.path.join("launch", "*.launch.py"))),
        (share_dir + "/launch/rviz", glob.glob(os.path.join("launch/rviz", "*.rviz"))),
        (
            share_dir + "/grip_assets/urdf/robots",
            glob.glob(os.path.join("grip_assets/urdf/robots", "*")),
        ),
        (
            share_dir + "/grip_assets/config",
            glob.glob(os.path.join("grip_assets/config", "*.yaml")),
        ),
        (
            share_dir,
            glob.glob("*.md"),
        ),
    ],
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "rgbd_camera_example = grip_examples.sensors.ex02_ros_rgbd_camera:main",
            "ros_arm_example = grip_examples.robot.ex04_ros_robot:main",
            "ros_robot_moveit = grip_examples.robot.ex05_ros_robot_moveit:main",
        ],
    },
)
