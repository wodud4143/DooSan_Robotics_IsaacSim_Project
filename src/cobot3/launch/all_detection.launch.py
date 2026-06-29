from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([

        Node(
            package='cobot3',
            executable='detection_L',
            name='detection_L',
            output='screen'
        ),

        Node(
            package='cobot3',
            executable='detection_R',
            name='detection_R',
            output='screen'
        ),

    ])
