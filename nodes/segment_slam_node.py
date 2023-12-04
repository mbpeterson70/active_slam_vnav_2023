#!/usr/bin/env python3

import numpy as np

# ROS imports
import rospy
import cv_bridge
import message_filters

# ROS msgs
import nav_msgs.msg as nav_msgs
import sensor_msgs.msg as sensor_msgs
import active_slam.msg as active_slam_msgs

from active_slam.segment_slam.segment_slam import SegmentSLAM

from utils import pose_msg_2_T, T_2_pose_msg

class SegmentSLAMNode():

    def __init__(self):

        # ros params
        self.K = np.array(rospy.get_param("~K")).reshape((3,3))
        self.distorition_params = np.array(rospy.get_param("~distorition_params"))
        
        # internal variables
        self.slam = SegmentSLAM(self.K, self.distorition_params)
        self.received_first_msg = False
        self.seen_objects = set() 
        
        # ros subscribers
        self.meas_sub = rospy.Subscriber("measurement_packet", active_slam_msgs.MeasurementPacket, callback=self.meas_packet_cb, queue_size=5)

        # ros publishers
        

    def meas_packet_cb(self, packet: active_slam_msgs.MeasurementPacket):
        """
        Called every time a new measurement packet is published
        """
        
        self.slam.add_relative_pose(
            pose_msg_2_T(packet.incremental_pose.pose), 
            packet.incremental_pose.covariance, 
            pre_idx=packet.sequence - 1
        )

        for segment in packet.segments:
            self.slam.add_segment_measurement(
                object_id=segment.id,
                measurement=segment.center,
                pixel_std_dev=segment.covariance,
                initial_guess=segment.initial_guess,
                pose_idx=segment.sequence
            )

        return

def main():

    rospy.init_node('segment_slam')
    node = SegmentSLAMNode()
    rospy.spin()

if __name__ == "__main__":
    main()
