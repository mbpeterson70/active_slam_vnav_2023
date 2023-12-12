#!/usr/bin/env python3

import numpy as np
import gtsam

# ROS imports
import rospy
import cv_bridge
import message_filters

# ROS msgs
import nav_msgs.msg as nav_msgs
import geometry_msgs.msg as geometry_msgs
import sensor_msgs.msg as sensor_msgs
import active_slam.msg as active_slam_msgs
import visualization_msgs.msg as visualization_msgs

from active_slam.segment_slam.segment_slam import SegmentSLAM

from utils import pose_msg_2_T, T_2_pose_msg

class SegmentSLAMNode():

    def __init__(self):

        # ros params
        self.K = np.array(rospy.get_param("~K")).reshape((3,3))
        self.distorition_params = np.array(rospy.get_param("~distorition_params"))
        self.T_BC = np.array(rospy.get_param("~T_BC", np.eye(4).tolist())).reshape((4,4))
        self.frame_id = rospy.get_param("~frame_id", "Multirotor")
        
        # internal variables
        self.slam = SegmentSLAM(self.K, self.distorition_params, self.T_BC)
        self.received_first_msg = False
        self.seen_objects = set() 
        self.badly_behaved_ids = []
        self.obj_colors = dict()
        
        # ros subscribers
        self.meas_sub = rospy.Subscriber("measurement_packet", active_slam_msgs.MeasurementPacket, callback=self.meas_packet_cb, queue_size=5)

        # ros publishers
        self.opt_path_pub = rospy.Publisher("optimized_path", nav_msgs.Path, queue_size=5)
        self.opt_obj_pub = rospy.Publisher("optimized_objects", visualization_msgs.MarkerArray, queue_size=5)
        self.graph_pub = rospy.Publisher("factor_graph", active_slam_msgs.Graph, queue_size=5)

    def meas_packet_cb(self, packet: active_slam_msgs.MeasurementPacket):
        """
        Called every time a new measurement packet is published
        """
        # TODO: the message setup does not make this super clear that the first incremental pose
        # is actually used as the initial pose.
        if packet.sequence == 0:
            self.slam.set_initial_pose(pose_msg_2_T(packet.incremental_pose.pose))
        else:
            covariance_tmp = np.array(packet.incremental_pose.covariance).reshape((6,6))
            covariance = np.zeros((6,6))
            covariance[3:,3:] = covariance_tmp[:3,:3]
            covariance[:3,:3] = covariance_tmp[3:,3:]
            self.slam.add_relative_pose(
                pose_msg_2_T(packet.incremental_pose.pose), 
                covariance, 
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
                print(f"Object {seg_id} triangulation failed! Ignoring...")
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
            disable_data_association=False
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

        for i in range(10):
            try:
                import time
                start_t = time.time()
                result, marginals = self.slam.solve()
                print(f"TOTAL TIME: {time.time() - start_t}")
                break
            except Exception as ex:
                # print(ex.args)
                self.handle_gtsam_exception(ex)
                # print(gtsam.VariableIndex(self.slam.graph))
                continue
        # print("GOT A RESULT")
        # print(result)
        # print(gtsam.VariableIndex(self.slam.graph))
        # print(gtsam.VariableIndex(self.slam.graph).find(self.slam.x(0)))
        # print(gtsam.VariableIndex(self.slam.graph).__dir__())

        self.publish_graph(result, marginals)
        self.publish_optimized_path(result)
        self.publish_optimized_objects(result)

        return
    
    def publish_graph(self, result, marginals):
        graph_msg = active_slam_msgs.Graph()
        for i in range(len(self.slam.pose_chain)):
            new_node = active_slam_msgs.GraphNode()
            new_node.id = active_slam_msgs.GraphNodeID(ord('x'), i)
            position = result.atPose3(self.slam.x(i)).matrix()[:3,3]
            new_node.position.x, new_node.position.y, new_node.position.z = position
            new_node.covariance = marginals.marginalCovariance(
                self.slam.x(i))[3:6,3:6].reshape(-1).tolist() # translation piece
            # TODO: send marginal covariance
            graph_msg.nodes.append(new_node)
            
        for obj_id in self.slam.object_ids:
            new_node = active_slam_msgs.GraphNode()
            new_node.id = active_slam_msgs.GraphNodeID(ord('o'), obj_id)
            position = result.atPoint3(self.slam.o(obj_id))
            new_node.position.x, new_node.position.y, new_node.position.z = position
            new_node.covariance = marginals.marginalCovariance(
                self.slam.o(obj_id)).reshape(-1).tolist()
            graph_msg.nodes.append(new_node)
            
        # populate edges
        i = 0
        num_factors = 0
        while num_factors < self.slam.graph.nrFactors():
            if self.slam.graph.exists(i):
                num_factors += 1
            else:
                i += 1
                continue
            factor = self.slam.graph.at(i)
            i += 1
            if len(factor.keys()) != 2:
                continue
            new_edge = active_slam_msgs.GraphEdge()
            for j, k in enumerate(factor.keys()):
                new_edge.edge[j] = self.key_to_id(k)
            graph_msg.edges.append(new_edge)           
        
        self.graph_pub.publish(graph_msg) 
    
    def publish_optimized_path(self, result):
        path = nav_msgs.Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = self.frame_id

        for i in range(len(self.slam.pose_chain)):
            pose = geometry_msgs.PoseStamped()
            pose.pose = T_2_pose_msg(result.atPose3(self.slam.x(i)).matrix())
            path.poses.append(pose)

        self.opt_path_pub.publish(path)
        return
    
    def publish_optimized_objects(self, result):
        marker_array = visualization_msgs.MarkerArray()
        for obj_id in self.slam.object_ids:
            marker_array.markers.append(self.get_object_marker(obj_id, result.atPoint3(self.slam.o(obj_id))))
        self.opt_obj_pub.publish(marker_array)
        
    def handle_gtsam_exception(self, ex):
        if "Indeterminant linear system" in ex.args[0]:
            variable = ex.args[0].split('Symbol: ')[1].split(').')[0].strip()
            if variable[0] == 'o':
                obj_num = int(variable[1:])
                rospy.logwarn(f"Object {obj_num} may have caused graph to become indeterminant. " + 
                              "Attempting to resolve without object.")
                deleted_ids = self.slam.remove_object(obj_num)
                self.badly_behaved_ids += deleted_ids
            else:
                raise ex
        else:
            raise ex
    
    def get_object_marker(self, obj_id, position):

        if obj_id in self.obj_colors:
            color = self.obj_colors[obj_id]
        else:
            color = np.random.random(3)
            self.obj_colors[obj_id] = color

        marker = visualization_msgs.Marker()
        marker = visualization_msgs.Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = self.frame_id
        marker.id = obj_id
        marker.type = marker.CUBE
        marker.action = marker.ADD
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.lifetime = rospy.Duration.from_sec(10.0)
        marker.frame_locked = 1
        marker.color.a = 1
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.pose =  geometry_msgs.Pose()

        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2]
        marker.pose.orientation.w = 1.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        return marker
    
    def key_to_id(self, key):
        symbol = gtsam.Symbol(key)
        return active_slam_msgs.GraphNodeID(symbol.chr(), symbol.index())

def main():

    rospy.init_node('segment_slam')
    node = SegmentSLAMNode()
    rospy.spin()

if __name__ == "__main__":
    main()