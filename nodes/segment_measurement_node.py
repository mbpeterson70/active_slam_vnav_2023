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


class SegmentMeasurementNode():

    def __init__(self):
        
        # internal variables
        self.bridge = cv_bridge.CvBridge()

        # ros params
        self.num_meas_new_obj = rospy.get_param("~num_meas_new_obj", 3) # number of objects needed 
                                                                        # to create new object, default: 3
        
        # ros subscribers
        subs = [
            message_filters.Subscriber("/airsim_node/Multirotor/odom_local_ned", nav_msgs.Odometry),
            message_filters.Subscriber("/airsim_node/Multirotor/front_center_custom/Scene",
                                       sensor_msgs.Image),
            message_filters.Subscriber("/airsim_node/Multirotor/front_center_custom/Scene/camera_info", 
                                       sensor_msgs.CameraInfo),
        ]
        self.ts = message_filters.ApproximateTimeSynchronizer(subs, queue_size=1, slop=.1)
        self.ts.registerCallback(self.cb) # registers incoming messages to callback

        # ros publishers
        self.meas_pub = rospy.Publisher("measurement_packet", active_slam_msgs.MeasurementPacket, queue_size=5)

    def cb(self, *msgs):
        """
        This function gets called every time synchronized odometry, image message, and camera info 
        message are available.
        """
        odom_msg, img_msg, cam_info_msg = msgs

        # conversion from ros msg to cv img
        img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')

        # extract camera intrinsics
        K = np.array([cam_info_msg.K]).reshape((3,3))

        # create packet
        packet = active_slam_msgs.MeasurementPacket()

        # TODO: fill in packet creation
        print(packet)

        self.meas_pub.publish(packet)

        return

def main():

    rospy.init_node('segment_measurement')
    node = SegmentMeasurementNode()
    rospy.spin()

if __name__ == "__main__":
    main()
