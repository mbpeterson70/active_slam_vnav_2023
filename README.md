# Active SLAM VNAV 2023

<a target="_blank" href=""><img src="./images/success.gif" width="330" height="300" alt="successful loop closure"></a>
<a target="_blank" href=""><img src="./images/active_slam_mapping.gif" width="300" height="300" alt="mapping"></a>

# Installation

### AirSim


1. Download environment binary (Linux): [https://github.com/Microsoft/AirSim/releases](https://github.com/Microsoft/AirSim/releases)
   1. After, unzip
(change this for different environments) Run: ``./AbandonedPark/LinuxNoEditor/AbandonedPark/Binaries/Linux/AbandonedPark -ResX=640 -ResY=480 -windowed``
1. Install ROS wrapper: [https://microsoft.github.io/AirSim/airsim_ros_pkgs/](https://microsoft.github.io/AirSim/airsim_ros_pkgs/)

### FastSAM Perception Frontend

Follow the instructions for installing FastSAM [here](https://github.com/CASIA-IVA-Lab/FastSAM). Make sure that FastSAM is on your python path! (You can run the below to do that)

export PYTHONPATH="$PYTHONPATH:<directory to your FastSAM install>

### `active_slam` Python Package

`cd` into this directory and `pip install .`

### `tmuxp`

```
sudo apt install tmuxp
```

# Setup

`cd` into this directory and run

```
source ./utils/cp_settings.sh
```

to setup the AirSim settings.

# Running the full system

Open up `tmux/perception_exploration.yaml`.
In the top of the file are a few environment parameters that you should modify to fit your environment. Alternatively you can just set those before running the tmux script (second option below)

Option 1:

```
tmuxp load ./tmux/perception_exploration.yaml 
```

Option 2:

```
AIRSIM_BIN_PATH=<path to AirSim binary> \
ACTIVE_SLAM_VENV=<path to root python virtual environment> \
ACTIVE_SLAM_WS=<path to ROS workspace> \
tmuxp load ./tmux/perception_exploration.yaml 
```

You can visualize by opening up rviz:

```
rviz -d ./rviz/active_slam.rviz
```

# Running Active Planner Alone

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
