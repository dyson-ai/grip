import os
import glob
from setuptools import setup, find_packages


package_name = "grip"
share_dir = os.path.join("share", package_name)

version_file = os.path.join(os.path.dirname(__file__), "grip/version.py")
with open(version_file, "r") as f:
    # use eval to get a clean string of version from file
    __version__ = eval(f.read().strip().split("=")[-1])


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

core_requirements = [
    "setuptools<=65",
    "numpy>=1.25.2",
    "scipy>=1.11.1",
    "pybullet>=3.2.4",
    "open3d>=0.10.0",
    "opencv-python>=4.9.0.80",
    "matplotlib>=3.3.4",
    # "ghalton==0.6.1",
    "pybullet_planning@git+https://github.com/eaa3/pybullet_planning.git@dev",
    "trimesh>=3.9.20",
    "xatlas>=0.0.7",
    "transforms3d==0.4.1",
    "strenum",
]

setup(
    name=package_name,
    version=__version__,
    description=(
        "Grip is a prototyping toolbox for manipulation research."
        "It contains a collection of tools for creating robot manipulation environments, loading arbitrary robots, including grippers, robot arms, sensors and optional ROS2 interfaces."
    ),
    url="https://github.com/dyson-ai/grip",
    author="Ermano Arruda",
    maintainer="Ermano Arruda",
    maintainer_email="ermano.arruda@dyson.com",
    license="MIT",
    install_requires=core_requirements,
    extras_require={
        "dev": [
            "pytest",
            "ipdb",
            "black>=23,<=23.3",
            "check-manifest>=0.49,<=0.49",
            "pre-commit>=3.3.3,<=3.3.3",
            "pylint>=2.16,<=2.17.5",
            "pytest-cov>=4.1,<=4.1",
            "pytest-mock>=3.10,<=3.11.1",
            "pytest-runner<=6.0,>=6.0",
            "pytest>=7.4,<=7.4",
            "hypothesis>=6.82,<=6.82.2",
            "ruff>=0.0.28,<=0.0.28",
            "coverage>=7.2.7,<=7.3.0",
            "sphinx",
        ],
    },
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
