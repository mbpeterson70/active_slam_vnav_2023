import rosbag
import csv

# Set the path to the ROS bag file
bag_path = "/home/ankj/data/active_slam_data/vel2_drone_2023-11-15-14-42-46.bag"

# Set the name of the topic to extract data from
topic_name = "/airsim_node/Multirotor/imu/Imu"

# Set the path to save the CSV file
csv_path = "/home/ankj/data/active_slam_data/vel2_drone_2023-11-15-14-42-46_bag.csv"

# Open the ROS bag file
bag = rosbag.Bag(bag_path)

# Create a CSV file and write the header row
with open(csv_path, "w") as csv_file:
  writer = csv.writer(csv_file)
  writer.writerow(["timestamp", "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z", "angular_velocity_x", "angular_velocity_y", "angular_velocity_z"])

  # Iterate through the messages in the specified topic
  for topic, msg, t in bag.read_messages(topics=[topic_name]):
    # Extract the data from the message
    timestamp = msg.header.stamp.to_sec()
    linear_acceleration_x = msg.linear_acceleration.x
    linear_acceleration_y = msg.linear_acceleration.y
    linear_acceleration_z = msg.linear_acceleration.z
    angular_velocity_x = msg.angular_velocity.x
    angular_velocity_y = msg.angular_velocity.y
    angular_velocity_z = msg.angular_velocity.z

    # Write the data to the CSV file
    writer.writerow([timestamp, linear_acceleration_x, linear_acceleration_y, linear_acceleration_z, angular_velocity_x, angular_velocity_y, angular_velocity_z])

# Close the ROS bag file
bag.close()
