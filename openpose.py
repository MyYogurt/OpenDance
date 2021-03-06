# To use Inference Engine backend, specify location of plugins:
# source /opt/intel/computer_vision_sdk/bin/setupvars.sh
import cv2 as cv
import numpy as np
import argparse
import math
import tkinter as tk
from tkinter import *
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo

parser = argparse.ArgumentParser(description='OpenDance')
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--proto', help='Path to .prototxt')
parser.add_argument('--model', help='Path to .caffemodel')
parser.add_argument('--dataset', help='Specify what kind of model was trained. '
                                      'It could be (COCO, MPI, HAND) depends on dataset.')
parser.add_argument('--thr', default=0.1, type=float, help='Threshold value for pose parts heat map')
parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')
parser.add_argument('--scale', default=0.003922, type=float, help='Scale for blob.')

args = parser.parse_args()

if args.dataset == 'COCO':
    BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                   "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                   "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                   "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

    POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                   ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                   ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                   ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                   ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]
elif args.dataset == 'MPI':
    BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                   "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                   "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                   "Background": 15 }

    POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                   ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                   ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                   ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]
else:
    assert(args.dataset == 'HAND')
    BODY_PARTS = { "Wrist": 0,
                   "ThumbMetacarpal": 1, "ThumbProximal": 2, "ThumbMiddle": 3, "ThumbDistal": 4,
                   "IndexFingerMetacarpal": 5, "IndexFingerProximal": 6, "IndexFingerMiddle": 7, "IndexFingerDistal": 8,
                   "MiddleFingerMetacarpal": 9, "MiddleFingerProximal": 10, "MiddleFingerMiddle": 11, "MiddleFingerDistal": 12,
                   "RingFingerMetacarpal": 13, "RingFingerProximal": 14, "RingFingerMiddle": 15, "RingFingerDistal": 16,
                   "LittleFingerMetacarpal": 17, "LittleFingerProximal": 18, "LittleFingerMiddle": 19, "LittleFingerDistal": 20,
                   }

    POSE_PAIRS = [ ["Wrist", "ThumbMetacarpal"], ["ThumbMetacarpal", "ThumbProximal"],
                   ["ThumbProximal", "ThumbMiddle"], ["ThumbMiddle", "ThumbDistal"],
                   ["Wrist", "IndexFingerMetacarpal"], ["IndexFingerMetacarpal", "IndexFingerProximal"],
                   ["IndexFingerProximal", "IndexFingerMiddle"], ["IndexFingerMiddle", "IndexFingerDistal"],
                   ["Wrist", "MiddleFingerMetacarpal"], ["MiddleFingerMetacarpal", "MiddleFingerProximal"],
                   ["MiddleFingerProximal", "MiddleFingerMiddle"], ["MiddleFingerMiddle", "MiddleFingerDistal"],
                   ["Wrist", "RingFingerMetacarpal"], ["RingFingerMetacarpal", "RingFingerProximal"],
                   ["RingFingerProximal", "RingFingerMiddle"], ["RingFingerMiddle", "RingFingerDistal"],
                   ["Wrist", "LittleFingerMetacarpal"], ["LittleFingerMetacarpal", "LittleFingerProximal"],
                   ["LittleFingerProximal", "LittleFingerMiddle"], ["LittleFingerMiddle", "LittleFingerDistal"] ]


inWidth = args.width
inHeight = args.height
inScale = args.scale

net = cv.dnn.readNet(cv.samples.findFile(args.proto), cv.samples.findFile(args.model))

totalFrames = 0


def processVideo(file_path):
    global totalFrames
    cap = cv.VideoCapture(file_path)
    totalFrames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    frame_points = []
    frames = []
    frame_num = 0

    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break

        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        inp = cv.dnn.blobFromImage(frame, inScale, (inWidth, inHeight),
                                   (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inp)
        out = net.forward()

        assert(len(BODY_PARTS) <= out.shape[1])

        points = []
        for i in range(len(BODY_PARTS)):
            # Slice heatmap of corresponding body's part.
            heatMap = out[0, i, :, :]

            # Originally, we try to find all the local maximums. To simplify a sample
            # we just find a global one. However only a single pose at the same time
            # could be detected this way.

            # Only works for one dancer on screen
            _, conf, _, point = cv.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]

            # Add a point if it's confidence is higher than threshold.
            # if args.thr:
            #   points.append((int(x), int(y)))
            points.append((int(x), int(y)) if conf > args.thr else None)

        # draw the body pose on the frame
        for pair in POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            assert(partFrom in BODY_PARTS)
            assert(partTo in BODY_PARTS)

            # gives an index into the points array
            idFrom = BODY_PARTS[partFrom]
            idTo = BODY_PARTS[partTo]


            if points[idFrom] and points[idTo]:
                cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
                cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

        frame_points.append(points)
        t, _ = net.getPerfProfile()
        freq = cv.getTickFrequency() / 1000
        # cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        print("Processed frame #"+str(frame_num))
        frame_num += 1
        frames.append(frame)
    return frames, frame_points


def computeAngleDiffs(points1, points2):
    validCount = 0
    angleDiff = 0
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        # gives index into the points array
        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        # if they are not None
        #if None in (points1[idFrom])
        epsilon = .00001
        if points1[idFrom] and points1[idTo] and points2[idFrom] and points2[idTo]:
            yDiff1 = points1[idTo][1] - points1[idFrom][1]
            xDiff1 = points1[idTo][0] - points1[idFrom][0]
            ratio1 = yDiff1/(xDiff1*1.0 + epsilon)
            angle1 = np.arctan2(1, 1/(ratio1 + epsilon))

            yDiff2 = points2[idTo][1] - points2[idFrom][1]
            xDiff2 = points2[idTo][0] - points2[idFrom][0]
            ratio2 = yDiff2/(xDiff2 * 1.0 + epsilon)
            angle2 = np.arctan2(1, 1/(ratio2 + epsilon))

            print("ANGLE")
            print(angle1, angle2)
            rawAngleDiff = abs(angle2 - angle1)
            validCount += 1
            angleDiff += min(rawAngleDiff, 2*math.pi - rawAngleDiff)
            print("AngleDiff:", angleDiff)
    # the acc is 1 - (our error/max possible error)
    accuracy = 1 - (angleDiff/(validCount*math.pi))
    return accuracy * 100

def computeAvg(running_avg, new_num, curr_length):
    if curr_length == 1:
        return new_num
    else:
        return running_avg * (curr_length - 1)/curr_length + new_num/curr_length

def process():
    if youtube_video == "" or my_video == "":
        showinfo(title="Error", message="You must have two videos selected")
        return
    ref_frames, ref_points = processVideo(youtube_video)
    print("processed vid1")
    test_frames, test_points = processVideo(my_video)
    print("processed vid2")
    # the point arrays are 2d arrays where unfound points in a frame are None
    print(len(ref_points), len(test_points))

    running_avg = 0
    for frame_num in range(min(len(ref_frames), len(test_frames))):
        frame_1 = ref_frames[frame_num]
        frame_2 = test_frames[frame_num]

        frame_acc = computeAngleDiffs(ref_points[frame_num], test_points[frame_num])
        running_avg = computeAvg(running_avg, frame_acc, frame_num + 1)

        print("FRAMEACC:", frame_acc)

        # in order to concatenate, both imgaes need to have the same width
        width_ratio = frame_1.shape[1] / frame_2.shape[1]
        if width_ratio < 1:
            cv.resize(frame_2, (0, 0), fx=width_ratio, fy=width_ratio)
        else:
            cv.resize(frame_1, (0, 0), fx=1 / width_ratio, fy=1 / width_ratio)

        print("DIMS:", width_ratio, frame_1.shape[1], frame_2.shape[1])
        im_concat = cv.vconcat([frame_1, frame_2])
        height = im_concat.shape[0]
        cv.putText(im_concat, '%.2f' % running_avg + "% ACC", (10, height - 10), cv.FONT_HERSHEY_SIMPLEX, 1,
                   (255, 255, 255))

        # cv.imshow('test', im_concat)
        success = cv.imwrite("./frames/" + str(frame_num).zfill(4) + '.png', im_concat)
        print("saving frame #", frame_num, success)

root = tk.Tk()
root.title('OpenDance')
root.geometry('500x200')

youtube_video = ""
my_video = ""
output_folder = ""

youtube_label_var = StringVar()
youtube_label_var.set("Your video to compare to:")
my_video_label_var = StringVar()
my_video_label_var.set("Your personal video:")
output_folder_label_var = StringVar()
output_folder_label_var.set("Ouput folder: ")

def select_file_correct_video():
    filetypes = [("Video files", ".mov .mp4")]

    global youtube_video
    youtube_video = fd.askopenfilename(
        title='Choose video',
        filetypes=filetypes
    )

    youtube_label_var.set("Your video to compare to: " + youtube_video)

def select_file_user_video():
    filetypes = [("Video files", ".mov .mp4")]

    global my_video
    my_video = fd.askopenfilename(
        title='Choose video',
        filetypes=filetypes
    )

    my_video_label_var.set("Your personal video: " + my_video)

def select_output_folder():
    global output_folder

    output_folder = fd.askdirectory(
        title='Choose output directory',
    )

    output_folder_label_var.set("Output folder: "+output_folder)

open_correct_button = Button(
    root,
    text='Choose music video input',
    command=select_file_correct_video
)

correct_label = Label(root, textvariable=youtube_label_var)
personal_label = Label(root, textvariable=my_video_label_var)
output_label = Label(root, textvariable=output_folder_label_var)

open_user_button = Button(
    root,
    text="Choose your video",
    command=select_file_user_video
)

output_button = Button(
    root,
    text='Choose output folder',
    command=select_output_folder
)

compute_button = Button(
    root,
    text='Compute',
    command=process
)

open_correct_button.pack()
correct_label.pack()
open_user_button.pack()
personal_label.pack()
output_button.pack()
output_label.pack()
compute_button.pack()


root.mainloop()