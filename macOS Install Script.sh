#!/bin/bash
git clone https://github.com/MyYogurt/OpenDance.git
cd OpenDance/
git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
cd openpose/
git submodule update --init --recursive --remote
cd ..
curl http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel -o openpose/models/pose/mpi/pose_iter_160000.caffemodel
bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
brew update
bash /scripts/osx/install_deps.sh