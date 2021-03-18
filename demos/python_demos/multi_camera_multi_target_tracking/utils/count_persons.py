import os
from pathlib import Path

import cv2 as cv
import numpy as np


def crop_persons(frame_number, frame, detections):
    for i, obj in enumerate(detections):
        left, top, right, bottom = obj.rect
        label = obj.label
        id = int(label.split(' ')[-1]
                 ) if isinstance(label, str) else int(label)

        if id >= 0:
            crop_person(frame_number, frame, left, top, right, bottom, id)


def crop_person(frame_number, frame, left, top, right, bottom, id):
    crop_img = frame[top:bottom, left:right]
    person_dir = f'/mnt/data/person/{id:08d}'
    Path(person_dir).mkdir(parents=True, exist_ok=True)
    cv.imwrite(os.path.join(
        person_dir, f'frame_{frame_number:08d}.jpg'), crop_img)


def count_persons(frame_number, frames, all_objects, fps='', show_all_detections=True,
                  max_window_size=(1920, 1080), stack_frames='vertical'):
    assert len(frames) == len(all_objects)
    assert stack_frames in ['vertical', 'horizontal']
    vis = None
    for i, (frame, objects) in enumerate(zip(frames, all_objects)):
        crop_persons(frame_number, frame, objects)
        if vis is not None:
            if stack_frames == 'vertical':
                vis = np.vstack([vis, frame])
            elif stack_frames == 'horizontal':
                vis = np.hstack([vis, frame])
        else:
            vis = frame
