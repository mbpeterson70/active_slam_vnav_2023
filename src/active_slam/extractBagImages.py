import rosbag
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

bag_file = "/home/ankj/data/active_slam_data/vel2_drone_2023-11-15-14-42-46.bag"
image_topic = "/airsim_node/Multirotor/front_center_custom/Scene"
output_folder = "/home/ankj/data/active_slam_data/BagImages"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

bridge = CvBridge()

with rosbag.Bag(bag_file, "r") as bag:
    for topic, msg, t in bag.read_messages(topics=[image_topic]):
        img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        filename = os.path.join(output_folder, str(t) + ".png")
        cv2.imwrite(filename, img)
