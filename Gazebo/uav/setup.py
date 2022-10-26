from setuptools import setup

package_name = 'uav'
path_following = 'uav/path_following'
collision_avoidance = 'uav/collision_avoidance'
path_planning = 'uav/path_planning'
setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name,  path_following, collision_avoidance, path_planning],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'Offboard = uav.offboard:main',
            'Ca = uav.ca:main',
            'Att = uav.pf_att:main',
            'Guid = uav.pf_guid:main',
            'Gpr = uav.pf_gpr:main'
        ],
    },
)
