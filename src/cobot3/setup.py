from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'cobot3'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/ament_index/resource_index/packages', ['resource/cobot3']),
        ('share/cobot3', ['package.xml']),
        (os.path.join('share', 'cobot3', 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='CodeBaro',
    maintainer_email='wodud4143@naver.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'detection_L = cobot3.detection_L:main',
            'detection_R = cobot3.detection_R:main',
        ],
    },
)
