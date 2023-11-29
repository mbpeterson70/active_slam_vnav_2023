import rosbag
import csv

bag_file = '/home/annika/data/high_up_lawn_mower.bag'
topic = '/airsim_node/Multirotor/odom_local_ned'

output_file = '/home/annika/data/AirsimGT.csv'

# Open the bag file
bag = rosbag.Bag(bag_file)

# Create a CSV file and write the headers
with open(output_file, 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['x', 'y', 'z', 'R'])

    # Iterate over the messages in the bag file
    for topic, msg, t in bag.read_messages(topics=[topic]):
        # Extract the ground truth pose and yaw
        # print(msg)
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        Rx = msg.pose.pose.orientation.x
        Ry = msg.pose.pose.orientation.y
        Rz = msg.pose.pose.orientation.z
        Rw = msg.pose.pose.orientation.w
        t = msg.header.stamp.to_sec()

        # Write the data to the CSV file
        writer.writerow([x, y, z, Rx, Ry, Rz, Rw, t])

# Close the bag file
bag.close()
