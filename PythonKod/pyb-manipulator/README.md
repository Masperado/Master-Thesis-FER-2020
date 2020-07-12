# pyb-manipulator
Pybullet wrapper which simplifies many manipulator-specific queries and commands.
All you need to get started is a URDF.

# Installation
Install the [liegroups](https://github.com/utiasSTARS/liegroups) package.

Clone this repository. To install, `cd` into the repository directory (the one with `setup.py`) and run:
```bash
pip install -e .
```

# Roadmap
- ~~Abstract away initializations for specific robots to separate classes~~
- Implement selectively damped least squares for enhanced stability
- Support for moving bases
- Separate gripper class to generalize grasping
- ~Support for `pip install`~
- ~~PI control~~
- Force-torque control




