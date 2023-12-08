import rosbag
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

#bag_file = "/home/ankj/data/active_slam_data/vel1_alt30m_airsimNH_drone_2023-11-16-16-29-57.bag"
bag_file = '/home/annika/data/high_up_lawn_mower.bag'
image_topic = "/airsim_node/Multirotor/front_center_custom/Scene"
#output_folder = "/home/ankj/data/active_slam_data/BagImages"
output_folder = '/home/annika/data/high_up_lawn_mower_images'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

bridge = CvBridge()

with rosbag.Bag(bag_file, "r") as bag:
    for topic, msg, t in bag.read_messages(topics=[image_topic]):
        img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        filename = os.path.join(output_folder, str(t) + ".png")
        #cv2.imwrite(filename, img)
        print(msg.header.stamp.to_sec())
