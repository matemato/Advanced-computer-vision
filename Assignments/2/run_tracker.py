from cProfile import label
import time

import matplotlib.pyplot as plt
import numpy as np
import cv2

from sequence_utils import VOTSequence
from ncc_tracker_example import NCCTracker, NCCParams
from ms_tracker import MSTracker, MSParams

fps = []
errors = []
params = [0.1, 0.5, 1, 3, 5, 9, 15, 30, 50]
params = [8, 16, 32, 64]
params = [0, 0.01, 0.05, 0.1, 0.25, 0.5]
params = [1]
for p in params:
    # set the path to directory where you have the sequences
    dataset_path = './Sequences' # set to the dataset path on your disk
    sequence = 'hand2'  # choose the sequence you want to test

    # visualization and setup parameters
    win_name = 'Tracking window'
    reinitialize = True
    show_gt = False
    video_delay = 15
    font = cv2.FONT_HERSHEY_PLAIN

    # create sequence object
    sequence = VOTSequence(dataset_path, sequence)
    init_frame = 0
    n_failures = 0
    # create parameters and tracker objects
    # parameters = NCCParams()
    # tracker = NCCTracker(parameters)

    parameters = MSParams(p)
    tracker = MSTracker(parameters)

    time_all = 0

    # initialize visualization window
    sequence.initialize_window(win_name)
    # tracking loop - goes over all frames in the video sequence
    frame_idx = 0
    while frame_idx < sequence.length():
        img = cv2.imread(sequence.frame(frame_idx))
        # initialize or track
        if frame_idx == init_frame:
            # initialize tracker (at the beginning of the sequence or after tracking failure)
            t_ = time.time()
            tracker.initialize(img, sequence.get_annotation(frame_idx, type='rectangle'))
            # print(sequence.get_annotation(frame_idx, type='rectangle'))
            time_all += time.time() - t_
            predicted_bbox = sequence.get_annotation(frame_idx, type='rectangle')
        else:
            # track on current frame - predict bounding box
            t_ = time.time()
            predicted_bbox = tracker.track(img)
            time_all += time.time() - t_

        # calculate overlap (needed to determine failure of a tracker)
        gt_bb = sequence.get_annotation(frame_idx, type='rectangle')
        o = sequence.overlap(predicted_bbox, gt_bb)

        # draw ground-truth and predicted bounding boxes, frame numbers and show image
        if show_gt:
            sequence.draw_region(img, gt_bb, (0, 255, 0), 1)
        sequence.draw_region(img, predicted_bbox, (0, 0, 255), 2)
        sequence.draw_text(img, '%d/%d' % (frame_idx + 1, sequence.length()), (25, 25))
        sequence.draw_text(img, 'Fails: %d' % n_failures, (25, 55))
        sequence.show_image(img, video_delay)

        if o > 0 or not reinitialize:
            # increase frame counter by 1
            frame_idx += 1
        else:
            # increase frame counter by 5 and set re-initialization to the next frame
            frame_idx += 5
            init_frame = frame_idx
            n_failures += 1

    print('Tracking speed: %.1f FPS' % (sequence.length() / time_all))
    print('Tracker failed %d times' % n_failures)
    fps.append((sequence.length() / time_all))
    errors.append(n_failures)

plt.style.use('seaborn-whitegrid')
plt.plot(np.array(params).astype('str'), errors, 'r', label="Normal")
plt.plot(np.array(params).astype('str'), errors, '.k')
plt.plot(np.array(params).astype('str'), [3,3,5,6,6,7], 'b', label="With background info")
plt.plot(np.array(params).astype('str'), [3,3,5,6,6,7], '.k')
plt.xticks(np.array(params).astype('str'))
plt.yticks(range(0,8))
plt.legend()
plt.title("Impact of alpha parameter")
plt.xlabel("Alpha")
plt.ylabel("Number of errors")
plt.show()

