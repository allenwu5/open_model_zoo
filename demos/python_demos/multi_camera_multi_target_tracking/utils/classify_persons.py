import numpy as np


def classify_persons(frame_time, frame, detections, classify_person_flow):
    person_class_dict = {}
    for i, obj in enumerate(detections):
        left, top, right, bottom = obj.rect
        label = obj.label
        id = int(label.split(' ')[-1]
                 ) if isinstance(label, str) else int(label)

        if id >= 0:
            crop_img = frame[top:bottom, left:right]

            x1y1x2y2 = (left, top, right, bottom)
            person_class, person_action = classify_person_flow(frame_time, id, crop_img, x1y1x2y2)
            person_class_dict[id] = (person_class, person_action)
    return person_class_dict


def classify_persons_per_frame(frame_times, frames, all_objects, classify_person_flow, fps='', show_all_detections=True,
                               max_window_size=(1920, 1080), stack_frames='vertical'):
    assert len(frames) == len(all_objects)
    assert stack_frames in ['vertical', 'horizontal']
    vis = None
    person_class_dict = {}
    for i, (frame_time, frame, objects) in enumerate(zip(frame_times, frames, all_objects)):
        person_class_dict.update(classify_persons(
            frame_time, frame, objects, classify_person_flow))
        if vis is not None:
            if stack_frames == 'vertical':
                vis = np.vstack([vis, frame])
            elif stack_frames == 'horizontal':
                vis = np.hstack([vis, frame])
        else:
            vis = frame
    return person_class_dict
