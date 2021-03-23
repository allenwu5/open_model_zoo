import numpy as np


def classify_persons(timestamp, frame, detections, classify_person_func):
    person_class_dict = {}
    for i, obj in enumerate(detections):
        left, top, right, bottom = obj.rect
        label = obj.label
        id = int(label.split(' ')[-1]
                 ) if isinstance(label, str) else int(label)

        if id >= 0:
            crop_img = frame[top:bottom, left:right]
            person_class = classify_person_func(timestamp, id, crop_img)
            person_class_dict[id] = person_class
    return person_class_dict


def classify_persons_per_frame(timestamps, frames, all_objects, classify_person_func, fps='', show_all_detections=True,
                               max_window_size=(1920, 1080), stack_frames='vertical'):
    assert len(frames) == len(all_objects)
    assert stack_frames in ['vertical', 'horizontal']
    vis = None
    person_class_dict = {}
    for i, (timestamp, frame, objects) in enumerate(zip(timestamps, frames, all_objects)):
        person_class_dict.update(classify_persons(
            timestamp, frame, objects, classify_person_func))
        if vis is not None:
            if stack_frames == 'vertical':
                vis = np.vstack([vis, frame])
            elif stack_frames == 'horizontal':
                vis = np.hstack([vis, frame])
        else:
            vis = frame
    return person_class_dict
