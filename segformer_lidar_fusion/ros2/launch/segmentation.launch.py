"""ROS 2 launch file for the SegFormer segmentation node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("engine_path", default_value="deploy/model.engine"),
        DeclareLaunchArgument("checkpoint_path", default_value=""),
        DeclareLaunchArgument("image_size", default_value="512"),
        DeclareLaunchArgument("num_classes", default_value="7"),
        DeclareLaunchArgument("camera_topic", default_value="/rover/camera/image_raw"),
        DeclareLaunchArgument("lidar_topic", default_value="/rover/lidar/points"),
        DeclareLaunchArgument("output_topic", default_value="/rover/segmentation/costmap"),

        Node(
            package="segformer_lidar_fusion",
            executable="segmentation_node",
            name="segformer_segmentation",
            parameters=[{
                "engine_path": LaunchConfiguration("engine_path"),
                "checkpoint_path": LaunchConfiguration("checkpoint_path"),
                "image_size": LaunchConfiguration("image_size"),
                "num_classes": LaunchConfiguration("num_classes"),
                "camera_topic": LaunchConfiguration("camera_topic"),
                "lidar_topic": LaunchConfiguration("lidar_topic"),
                "output_topic": LaunchConfiguration("output_topic"),
            }],
            output="screen",
        ),
    ])
