## Install
Create a virtual env with python3.11, activate it, and then:
```bash
apt install build-essentials
apt install libosmesa6-dev libgl1-mesa-glx libglfw3
cp -r mjpro150 $HOME/.mujoco/
pip install 'cython<3'
pip install mujoco-py==1.50.1
```
For spinningup:
```bash
sudo apt-get update && sudo apt-get install libopenmpi-dev
```