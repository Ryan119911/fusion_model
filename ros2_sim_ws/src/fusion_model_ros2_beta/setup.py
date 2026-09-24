from glob import glob

from setuptools import find_packages, setup

package_name = "fusion_model_ros2_beta"

setup(
    name=package_name,
    version="0.5.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        (
            "share/" + package_name + "/launch",
            [
                "launch/ur10_brush_ros2_beta.launch.py",
                "launch/ur3_brush_ros2_beta.launch.py",
                "launch/ur5_brush_ros2_beta.launch.py",
            ],
        ),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/urdf", glob("urdf/*.urdf")),
        ("share/" + package_name + "/rviz", ["rviz/ur10_brush.rviz"]),
        (
            "share/" + package_name + "/meshes/ur10/visual",
            glob("meshes/ur10/visual/*"),
        ),
        (
            "share/" + package_name + "/meshes/ur10/collision",
            glob("meshes/ur10/collision/*"),
        ),
        (
            "share/" + package_name + "/third_party/ur_description",
            glob("third_party/ur_description/*"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="robot",
    maintainer_email="robot@example.com",
    description="Factory-calibrated UR10 CB3 brush simulation with V16 six-field inversion.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "brush_trajectory_driver = fusion_model_ros2_beta.brush_trajectory_driver:main",
        ],
    },
)
