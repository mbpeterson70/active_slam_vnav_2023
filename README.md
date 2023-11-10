AirSim Installation
Create file ~/Documents/AirSim/settings.json (example below)
Download environment binary (Linux): https://github.com/Microsoft/AirSim/releases
After, unzip
(change this for different environments) Run: ./AbandonedPark/LinuxNoEditor/AbandonedPark/Binaries/Linux/AbandonedParks -ResX=640 -ResY=480 -windowed
Install ROS wrapper: https://microsoft.github.io/AirSim/airsim_ros_pkgs/
Once the sim is running, and ros environment is source (setup.bash)
roslaunch airsim_ros_pkgs airsim_node.launch 
roslaunch airsim_ros_pkgs rviz.launch
Test driving (rostopic pub /airsim_node/Car/car_cmd airsim_ros_pkgs/CarControls "{throttle: 1.0}" -r 10)
Can look at camera topic using rviz

settings.json

{
  "SettingsVersion": 1.2,
  "SimMode": "Car",
  "Vehicles": {
    "Car": {
      "VehicleType": "PhysXCar",
      "Cameras": {
        "front_center": {
          "CaptureSettings": [
            {
              "ImageType": 0,
              "Width": 672,
              "Height": 376,
              "FOV_Degrees": 90
            }
          ],
          "X": 1.0, "Y": 0.0, "Z": -1.5,
          "Pitch": 0.0, "Roll": 0.0, "Yaw": 0.0
        }
      }
    }
  }
}

