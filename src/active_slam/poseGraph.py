import csv
import os
import gtsam
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def plot_xyz_over_time(result, imu_data):
    # Extract timestamps and poses
    timestamps = [data[0] for data in imu_data]
    x = []
    y = []
    z = []
    for i in range(len(imu_data)):
        pose = result.atPose3(i+1)
        x.append(pose.x())
        y.append(pose.y())
        z.append(pose.z())

    # Create subplots for x, y, z
    fig, axs = plt.subplots(3)
    fig.suptitle('X, Y, Z over time')
    axs[0].plot(timestamps, x)
    axs[0].set(ylabel='X')
    axs[1].plot(timestamps, y)
    axs[1].set(ylabel='Y')
    axs[2].plot(timestamps, z)
    axs[2].set(xlabel='Time', ylabel='Z')

    plt.show()

def calculate_delta_pose(prev_pose, acc, gyro, dt):
    # Initialize delta_pose to the previous pose
    delta_pose = prev_pose

    # Calculate rotation from gyroscope readings
    rotation = R.from_rotvec(gyro * dt).as_matrix()

    # Update orientation part of delta_pose
    delta_pose = delta_pose.compose(gtsam.Pose3(gtsam.Rot3(rotation), gtsam.Point3(0, 0, 0)))

    # Calculate translation from accelerometer readings
    translation = 0.5 * acc * dt**2

    # Update position part of delta_pose
    delta_pose = delta_pose.compose(gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(*translation)))

    return delta_pose

# Parse IMU data from CSV file 
imu_data = []
with open('/home/ankj/data/active_slam_data/vel2_drone_2023-11-15-14-42-46_bag.csv', 'r') as f:
  reader = csv.reader(f)
  next(reader)  # skip header row
  for i, row in enumerate(reader, start=2):  # start=2 to account for header row
    try:
      timestamp = float(row[0])
      acc = np.array([float(row[1]), float(row[2]), float(row[3])])
      gyro = np.array([float(row[4]), float(row[5]), float(row[6])])
      imu_data.append((timestamp, acc, gyro))
    except Exception as e:
      print(f"Error on line {i}: {e}")

# length of imu data:
print("length of imu data: ", len(imu_data))

# print the accelerometer readings over time on a plot
timestamps = [data[0] for data in imu_data]
acc_x = [data[1][0] for data in imu_data]
acc_y = [data[1][1] for data in imu_data]
acc_z = [data[1][2] for data in imu_data]

# plot every 10th point
timestamps = timestamps[::10]
acc_x = acc_x[::10]
acc_y = acc_y[::10]
acc_z = acc_z[::10]
fig, axs = plt.subplots(3)
fig.suptitle('Accelerometer readings over time')
axs[0].plot(timestamps, acc_x)
axs[0].set(ylabel='X')
axs[1].plot(timestamps, acc_y)
axs[1].set(ylabel='Y')
axs[2].plot(timestamps, acc_z)
axs[2].set(xlabel='Time', ylabel='Z')

# Add image file names to list
image_files = []
for filename in os.listdir('/home/ankj/data/active_slam_data/BagImages'):
    if filename.endswith('.png'):
        image_files.append(filename)

print("Number of timestamps: ", len(imu_data))

# Sort image file names by timestamp
image_files.sort(key=lambda x: float(os.path.splitext(x)[0]))

# Create GTSAM graph
graph = gtsam.NonlinearFactorGraph()
initial_estimate = gtsam.Values()
imuNoiseModel = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))

# Add prior factor for first pose
pose = gtsam.Pose3()
graph.add(gtsam.PriorFactorPose3(1, pose, gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))))
initial_estimate.insert(1, pose)

# Define gravity
gravity = 9.81  # m/s^2

# Loop through IMU data and add relative pose factors to graph
poses = []
prev_pose = pose
for i in range(len(imu_data)):
    timestamp, acc, gyro = imu_data[i]
    
    # Subtract gravity from z-axis accelerometer reading
    acc[2] = -acc[2] 
    acc[2] -= gravity
    
    dt = imu_data[i+1][0] - timestamp if i < len(imu_data)-1 else 1.0/50.0  # assume 50 Hz if last data point
    delta_pose = calculate_delta_pose(prev_pose, acc, gyro, dt)
    pose = prev_pose.compose(delta_pose)
    poses.append(pose)
    prev_pose = pose

# Plot poses over time
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = [pose.x() for pose in poses]
y = [pose.y() for pose in poses]
z = [pose.z() for pose in poses]

# print number of poses
print("Number of poses: ", len(poses))

ax.scatter(x, y, z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
