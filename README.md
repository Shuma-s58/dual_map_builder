# dual_map_builder


```!
ros2 launch slam_toolbox online_async_launch.py slam_params_file:=slam_toolbox_params.yaml
```
```!
ros2 bag play tudanuma_test2_2026/ --clock --topics /low_scan /up_scan /tf /tf_static
```

```!
ros2 run dual_map_builder dual_map_node --ros-args -p scan_top_topic:=/up_scan -p base_frame:=base_link -p use_sim_time:=true
```
