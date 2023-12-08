import rosbag
from sensor_msgs.msg import CameraInfo

bag_file = '/home/annika/data/high_up_lawn_mower.bag'
topic_name = '/airsim_node/Multirotor/front_center_custom/Scene/camera_info'

# Open the rosbag file
bag = rosbag.Bag(bag_file)

# Iterate over the messages in the specified topic
for topic, msg, t in bag.read_messages(topics=[topic_name]):
    print("k matrix:", msg.K)
    print("p matrix:", msg.P)
    print("d matrix:", msg.D)
    # print(msg.K)
    # print(msg.D)
    # print(msg.R)
    # print(msg.P)

# Close the rosbag file
bag.close()
