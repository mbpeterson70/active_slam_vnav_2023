# AirSim Installation

1. Create file `~/Documents/AirSim/settings.json` (example below)
2. Download environment binary (Linux): [https://github.com/Microsoft/AirSim/releases](https://github.com/Microsoft/AirSim/releases)
   1. After, unzip
(change this for different environments) Run: ``./AbandonedPark/LinuxNoEditor/AbandonedPark/Binaries/Linux/AbandonedPark -ResX=640 -ResY=480 -windowed``
3. Install ROS wrapper: [https://microsoft.github.io/AirSim/airsim_ros_pkgs/](https://microsoft.github.io/AirSim/airsim_ros_pkgs/)
   1. Once the sim is running, and ros environment is sourced (setup.bash)
   2. `roslaunch airsim_ros_pkgs airsim_node.launch `
   3. `roslaunch airsim_ros_pkgs rviz.launch`

## other 

Test driving `(rostopic pub /airsim_node/Car/car_cmd airsim_ros_pkgs/CarControls "{throttle: 1.0}" -r 10)`

Can look at camera topic using rviz

## settings.json

```json
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
```

# AirSim simulation

1. start rosmaster (``roscore``)
2. start AirSim env of your choice (``./AbandonedPark/LinuxNoEditor/AbandonedPark/Binaries/Linux/AbandonedPark -ResX=640 -ResY=480 -windowed``)
3. start ros wrapper for AirSim (``roslaunch airsim_ros_pkgs airsim_node.launch``)
4. start planner (``roslaunch active slam planner_ros_node.launch``)

### if you want to change parameters

* you can pass params in planner (``roslaunch airsim_ros_pkgs airsim_node.launch car_or_drone:=drone altitude:=50.0 coverage_area_size:=100.0 exploration_goal_points_resolution:=10.0 exploration_velocity:=5.0``)
* exposed parames
  * ``car_or_drone``: vehicle type (currently car is not supported)
  * ``altitude``: exploratoin altitude
  * ``coverage_area_size``: coverage area size (currenty only square)
  * ``exploration_goal_points_resolution``: distance between goal points in exploration
  * ``exploration_velocity``: velocity constraint in exploration