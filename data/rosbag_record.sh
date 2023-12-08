#!/bin/bash

rosbag record   -o bags/drone \
		        /rosout \
                /rosout_agg \
                /tf \
                /tf_static \
                /airsim_node/Multirotor/global_gps \
                /airsim_node/Multirotor/gps/gps \
                /airsim_node/Multirotor/imu/Imu \
                /airsim_node/Multirotor/lidar/LidarCustom \
                /airsim_node/Multirotor/magnetometer/magnetometer \
                /airsim_node/Multirotor/odom_local_ned \
                /airsim_node/Multirotor/vel_cmd_body_frame \
                /airsim_node/Multirotor/vel_cmd_world_frame \
                /airsim_node/Multirotor/front_center_custom/Scene
            