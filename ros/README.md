## Install dependancies

Install `requiements_jetson.txt` file.

```bash
pip3 install -r requirements_jetson.txt
```

Build torchvision from source

https://zenn.dev/aung_yu/articles/aa1362fab01b0a


## Install ROS and dependencies

1. ROS install from source https://wiki.ros.org/Installation/Source
2. Intel Realsense Camera Driver install https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_jetson.md
3. Clone realsense library and ddynamic_reconfigure, and build them in the catkin workspace
```sh
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone https://github.com/IntelRealSense/realsense-ros.git
cd realsense-ros/
git checkout `git tag | sort -V | grep -P "^2.\d+\.\d+" | tail -1`
cd ..
git clone https://github.com/pal-robotics/ddynamic_reconfigure.git
cd ~/catkin_ws
catkin_make
```
4. Copy [om_teleop](eipl/eipl/tutorials/open_manipulator/ros/om_teleop) to `~/catkin_ws/src`
`eipl/eipl/tutorials/open_manipulator/ros/om_teleop` is a ROS package for controlling OpenManipulator.
```sh
cp -r eipl/eipl/tutorials/open_manipulator/ros/om_teleop ~/catkin_ws/src
cd ~/catkin_ws
catkin_make
```

