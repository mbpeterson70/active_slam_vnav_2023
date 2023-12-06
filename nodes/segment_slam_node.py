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
        self.badly_behaved_ids = []
        
        # ros subscribers
        self.meas_sub = rospy.Subscriber("measurement_packet", active_slam_msgs.MeasurementPacket, callback=self.meas_packet_cb, queue_size=5)

        # ros publishers
        

    def meas_packet_cb(self, packet: active_slam_msgs.MeasurementPacket):
        """
        Called every time a new measurement packet is published
        """
        
        self.slam.add_relative_pose(
            pose_msg_2_T(packet.incremental_pose.pose), 
            np.array(packet.incremental_pose.covariance).reshape((6,6)), 
            pre_idx=packet.sequence - 1
        )

        # Collect new segments and add existing ids
        new_segments = {}
        for segment in packet.segments:
            if segment.id not in self.slam.object_id_mapping:
                if segment.id in self.badly_behaved_ids:
                    continue
                if segment.id in new_segments:
                    new_segments[segment.id].append(segment)
                else:
                    new_segments[segment.id] = [segment]
                continue
            self.slam.add_segment_measurement(
                object_id=segment.id,
                center_pixel=np.array([segment.center.x, segment.center.y]),
                pixel_std_dev=np.array(segment.covariance).reshape((2,2)).diagonal()[0],
                # initial_guess=segment.initial_guess,
                pose_idx=segment.sequence
            )

        # Create initial guesses for new segments
        init_guesses = {}
        to_delete = []
        for seg_id, segment_measurements in new_segments.items():
            assert len(segment_measurements) >= 3, "not enough initial measurements received"
            try:
                init_guesses[seg_id] = self.slam.triangulate_object_init_guess(
                    pixels=[np.array([sm.center.x, sm.center.y]) for sm in segment_measurements],
                    pixel_std_dev=segment_measurements[0].covariance[0], # just takes the first covariance, TODO: change
                    pose_idxs=[sm.sequence for sm in segment_measurements]
                )
            except:
                # print([np.array([sm.center.x, sm.center.y]) for sm in segment_measurements])
                # print([sm.sequence for sm in segment_measurements])
                # for seq in [sm.sequence for sm in segment_measurements]:
                #     print(self.slam.pose_chain[seq])
                # init_guesses[seg_id] = np.zeros(3)
                self.badly_behaved_ids.append(seg_id)
                to_delete.append(seg_id)
        for seg_id in to_delete:
            del new_segments[seg_id]

        # Perform data association and add new segment measurements
        self.slam.new_objects_data_association(
            object_ids=[seg_id for seg_id in init_guesses],
            init_guesses=[init_guesses[seg_id] for seg_id in init_guesses],
            last_pose_idxs=[np.max([sm.sequence for sm in new_segments[seg_id]]) for seg_id in init_guesses]
        )      

        for seg_id, segment_measurements in new_segments.items():
            for sm in segment_measurements:
                self.slam.add_segment_measurement(
                    object_id=seg_id,
                    center_pixel=np.array([sm.center.x, sm.center.y]),
                    pixel_std_dev=sm.covariance[0], #not using full cov TODO change to a single float param
                    initial_guess=init_guesses[seg_id],
                    pose_idx=sm.sequence
                )

        result = self.slam.solve()
        print("GOT A RESULT")
        print(result)

        return

def main():

    rospy.init_node('segment_slam')
    node = SegmentSLAMNode()
    rospy.spin()

if __name__ == "__main__":
    main()
