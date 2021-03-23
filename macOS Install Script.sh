#!/bin/bash
git clone https://github.com/MyYogurt/OpenDance.git
cd OpenDance/
git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose
cd openpose/
git submodule update --init --recursive --remote
bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
brew update
bash /scripts/osx/install_deps.sh