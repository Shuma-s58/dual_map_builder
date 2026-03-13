from setuptools import setup
import os
from glob import glob

package_name = 'dual_map_builder'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your@email.com',
    description='Dual LiDAR map builder using slam_toolbox',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'dual_map_node = dual_map_builder.dual_map_node:main',
        ],
    },
)
