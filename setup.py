from setuptools import setup
import os
from glob import glob

package_name = 'yolo_meter_detection'

setup(
  name=package_name,
  version='0.0.1',
  packages=[
    package_name,
    f'{package_name}.nodes',
    f'{package_name}.utils',
    'yolo_meter_detection',
    'yolo_meter_detection.nodes',
    'yolo_meter_detection.utils'
  ],
  data_files=[
     ('share/ament_index/resource_index/packages', [
        'resource/' + package_name
     ]),
     ('share/' + package_name, ['package.xml']),
     ('share/' + package_name, ['codec.txt']),
     (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
     (os.path.join('share', package_name, 'cfg'), glob('cfg/*.yaml')),
     (os.path.join('share', package_name, 'rknnModel'), glob('rknnModel/*.rknn')),

  ],
  install_requires=['setuptools'],
  zip_safe=True,
  maintainer='Your Name',
  maintainer_email='your@email.com',
  description='YOLO-based meter detection pipeline for ROS2',
  license='Apache 2.0',
  tests_require=['pytest'],
  entry_points={
     'console_scripts': [
          'meter_detection_node = yolo_meter_detection.nodes.meter_detection_node:main',
      ],
  },
)
