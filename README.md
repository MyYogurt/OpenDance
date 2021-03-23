# Open Dance

## Inspiration

When we are hanging out with our friends, we want to dance, and since not many people have Just Dance or a video game console, it has been a struggle to find an activity to do that involved dancing. So we wanted to create a dancing game that would allow users to download and play for free, and also have the ability to choose any song instead of choosing from a limited selection.

## What it does

Open Dance allows users to search any music video on the internet and dance to it while earning a score at the end of the song based on the accuracy of the dancing.

## How we built it

OpenDance runs on the [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) technology developed by the Carnegie Mellon Perceptual Computing Lab. OpenPose is a real-time system to detect human keypoints on images and videos.

## Running

To run:

Ensure you meet the preqrequistes to run OpenPose. They can be found [here](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/1_prerequisites.md)

macOS users who have dveloped command line tools already installed can get all dependencies and code by running:

`curl -s https://raw.githubusercontent.com/MyYogurt/OpenDance/main/macOS%20Install%20Script.sh | bash`

Once all code and dependencies are installed run:

`python openpose.py --dataset MPI --model ./openpose/models/pose/mpi/pose_iter_160000.caffemodel --proto ./openpose/models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt` 

![Screenshot](screenshot.png)