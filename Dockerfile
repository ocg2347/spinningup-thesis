FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    build-essential \
    libosmesa6-dev libgl1-mesa-glx libglfw3

# Set the working directory
WORKDIR /app

# Create a virtual environment
RUN python3.11 -m venv venv

RUN . venv/bin/activate && pip install 'cython<3'

# Copy mjpro150 folder to $HOME/.mujoco
COPY .mujoco /root/.mujoco