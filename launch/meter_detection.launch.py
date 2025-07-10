import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
     package_name = 'yolo_meter_detection'

# 获取包共享目录
     package_share = get_package_share_directory(package_name)

# 模型路径
     model_path = os.path.join(package_share, 'rknnModel')
     return LaunchDescription([
        Node(
            package='yolo_meter_detection',
            executable='meter_detection_node',
            name='meter_detection_node',
            output='screen',
            parameters=[
               {'camera_topic': '/camera/image_raw'},
               {'video_device': '/dev/video21'},
               {'output_image_topic': '/meter_detection/image'},
               {'value_topic': '/meter_detection/value'},
               {'model_path': model_path}
            ]
        )
     ])
