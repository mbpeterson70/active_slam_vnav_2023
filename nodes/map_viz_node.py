#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, CameraInfo
from active_slam.msg import Graph
from cv_bridge import CvBridge, CvBridgeError
import cv2
from message_filters import TimeSynchronizer, Subscriber
from std_msgs.msg import Header
import numpy as np
from collections import deque
from nav_msgs.msg import Path

from utils import pose_msg_2_T, transform, xyz_2_pixel

class MapViz:
    def __init__(self):
        rospy.init_node('map_viz', anonymous=True)

        # Define parameters
        self.image_topic = 'airsim_node/Multirotor/front_center_custom/Scene'
        self.cam_info_topic = 'airsim_node/Multirotor/front_center_custom/Scene/camera_info'
        self.graph_topic = 'factor_graph'
        self.optimized_pose_topic = 'optimized_path'
        self.publish_topic = 'stitched_map'

        # rosparams
        self.overview_height = rospy.get_param("~cam_height", 1000)
        self.area_dist = rospy.get_param("~area_dist", 300)
        self.num_pixels = rospy.get_param("~num_pixels", 1500)
        self.K = np.array([
            [self.overview_height*self.num_pixels/self.area_dist, 0., .5*self.num_pixels],
            [0., self.overview_height*self.num_pixels/self.area_dist, .5*self.num_pixels],
            [0., 0., 1.]
        ])
        self.T_overview = np.eye(4)
        self.T_overview[2,3] = -self.overview_height
        self.overview_img = np.zeros((self.num_pixels,self.num_pixels,3)).astype(np.float64)
        self.num_imgs_added = 1
        self.seen_pixel_counts = np.ones((self.num_pixels,self.num_pixels,3))

        # Initialize ROS publishers and subscribers
        self.image_sub = Subscriber(self.image_topic, Image)
        self.cam_info_sub = Subscriber(self.cam_info_topic, Image)
        self.graph_sub = Subscriber(self.graph_topic, Graph)
        self.path_sub = Subscriber(self.optimized_pose_topic, Path)
        self.ts = TimeSynchronizer([self.image_sub, self.graph_sub, self.path_sub], 300)
        self.ts.registerCallback(self.callback)
        # TimeSynchronizer([self.image_sub, self.cam_info_sub], 100).registerCallback(lambda x, y : print(0))
        # TimeSynchronizer([self.image_sub, self.graph_sub,], 100).registerCallback(lambda x, y : print(1)) #
        # TimeSynchronizer([self.image_sub, self.path_sub], 100).registerCallback(lambda x, y : print(2)) #
        # TimeSynchronizer([self.cam_info_sub, self.graph_sub], 100).registerCallback(lambda x, y : print(3))
        # TimeSynchronizer([self.cam_info_sub, self.path_sub], 100).registerCallback(lambda x, y : print(4))
        # TimeSynchronizer([self.graph_sub, self.path_sub], 100).registerCallback(lambda x, y : print(5)) #

        self.image_pub = rospy.Publisher(self.publish_topic, Image, queue_size=10)

        # Initialize CvBridge
        self.bridge = CvBridge()

    def callback(self, image_msg: Image, graph_msg: Graph, path_msg: Path):
        K_cam = np.array([320.0, 0.0, 320.0, 0.0, 320.0, 240.0, 0.0, 0.0, 1.0]).reshape((3,3))
        T_cam = pose_msg_2_T(path_msg.poses[-1].pose)
        T_overview_cam = np.linalg.inv(self.T_overview) @ T_cam
        n_world = np.array([[0., 0., -1.]]).T
        n_cam = np.linalg.inv(T_cam)[:3,:3] @ n_world
        d = np.abs(T_cam[2,3])
        Hab = (T_overview_cam)[:3,:3] - (T_overview_cam)[:3,3].reshape((3,1)) @ n_cam.T / d
        # print(Hab)
        # # Hab = self.T_overview[:3,3].reshape((1,3)) @ self.T_overview[:3,:3] @ T_cam[:3,:3].T @ (-T_cam[:3,3]).reshape((3,1))
        
        # R_c2_w = self.T_overview[:3,:3].T
        # R_c1_w = T_cam[:3,:3].T
        # t_c2_w = np.linalg.inv(self.T_overview)[:3,3].reshape((3,1))
        # t_c1_w = np.linalg.inv(T_cam)[:3,3].reshape((3,1))

        # R21 = R_c2_w @ R_c1_w.T
        # t21 = R_c2_w @ (-R_c1_w.T @ t_c1_w) + t_c2_w
        # Hab = R21 - t21 @ n_cam.T / d
        # print(Hab)

        H = self.K @ Hab @ np.linalg.inv(K_cam)

        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # self.overview_img = self.overview_img*self.num_imgs_added/(self.num_imgs_added+1) + \
        #     cv2.warpPerspective(cv_image, H, self.overview_img.shape[1::-1])/(self.num_imgs_added+1)
        # self.overview_img = self.overview_img + \
        #     cv2.warpPerspective(cv_image, H, self.overview_img.shape[1::-1])*.2
        warped_img = cv2.warpPerspective(cv_image, H, self.overview_img.shape[1::-1]).astype(np.float64)
        new_seen_pixel_counts = self.seen_pixel_counts + np.where(warped_img != 0, 1, 0).astype(np.float64)
        self.overview_img = self.overview_img * self.seen_pixel_counts / new_seen_pixel_counts \
            + warped_img / new_seen_pixel_counts
        self.overview_img = self.overview_img.clip(0., 255.)
        self.num_imgs_added += 1
        
        overview = self.overview_img.astype(np.uint8)
        object_positions, object_ids = self.graph_2_object_positions(graph_msg)
        if len(object_ids) != 0:
            overview = self.draw_objects(object_positions, object_ids, overview)

        # Publish  image
        img_msg = self.bridge.cv2_to_imgmsg(overview, "bgr8")
        img_msg.header = Header(stamp=img_msg.header.stamp)
        self.image_pub.publish(img_msg)

    def graph_2_object_positions(self, graph_msg: Graph):
        object_positions = []
        object_ids = []
        for node in graph_msg.nodes:
            if node.id.type == ord('o'):
                object_positions.append([node.position.x, node.position.y, node.position.z])
                object_ids.append(node.id.index)
        return np.array(object_positions), np.array(object_ids)

    def draw_objects(self, object_positions, object_ids, img):
        obj_c = transform(np.linalg.inv(self.T_overview), object_positions, stacked_axis=0)
        obj_pixel = xyz_2_pixel(obj_c, self.K)

        # Draw segments on the image
        for o, oid in zip(obj_pixel, object_ids):
            center = (int(o[0]), int(o[1]))
            cv2.circle(img, center, 5, (0, 255, 0), -1)
            cv2.putText(img, str(oid), (center[0] + 10, center[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return img


    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = MapViz()
        node.run()
    except rospy.ROSInterruptException:
        pass