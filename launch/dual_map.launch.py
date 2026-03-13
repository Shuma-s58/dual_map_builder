from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    args = [
        DeclareLaunchArgument('bag_path',         description='rosbag のパス（必須）'),
        DeclareLaunchArgument('slam_params_file',  description='slam_toolbox params yamlのパス'),
        DeclareLaunchArgument('scan_top_topic',    default_value='/scan_top'),
        DeclareLaunchArgument('base_frame',        default_value='base_link'),
        DeclareLaunchArgument('map_resolution',    default_value='0.05'),
        DeclareLaunchArgument('map_size',          default_value='2000'),
        DeclareLaunchArgument('save_dir',          default_value='/tmp/dual_maps'),
        DeclareLaunchArgument('bag_rate',          default_value='1.0'),
        DeclareLaunchArgument('use_sim_time',      default_value='true'),
    ]

    # slam_toolboxはyaml内のscan_topicで/low_scanを受け取り、/mapと/tfを生成
    slam_toolbox_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('slam_toolbox'), 'launch', 'online_async_launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time':     LaunchConfiguration('use_sim_time'),
            'slam_params_file': LaunchConfiguration('slam_params_file'),
        }.items(),
    )

    dual_map_node = Node(
        package='dual_map_builder',
        executable='dual_map_node',
        name='dual_map_node',
        output='screen',
        parameters=[{
            'use_sim_time':    LaunchConfiguration('use_sim_time'),
            'scan_top_topic':  LaunchConfiguration('scan_top_topic'),
            'base_frame':      LaunchConfiguration('base_frame'),
            'map_resolution':  LaunchConfiguration('map_resolution'),
            'map_size':        LaunchConfiguration('map_size'),
            'save_map':        True,
            'save_dir':        LaunchConfiguration('save_dir'),
        }],
    )

    rosbag_play = TimerAction(
        period=3.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'ros2', 'bag', 'play',
                    LaunchConfiguration('bag_path'),
                    '--clock',
                    '-r', LaunchConfiguration('bag_rate'),
                ],
                output='screen',
            )
        ],
    )

    return LaunchDescription([*args, slam_toolbox_launch, dual_map_node, rosbag_play])
