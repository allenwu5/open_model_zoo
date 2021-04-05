"""
 Copyright (c) 2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
from .misc import COLOR_PALETTE


def draw_detections(frame, detections, person_class_dict, show_all_detections=True):
    """Draws detections and labels"""
    for i, obj in enumerate(detections):
        left, top, right, bottom = obj.rect
        label = obj.label
        id = int(label.split(' ')[-1]) if isinstance(label, str) else int(label)
        box_color = COLOR_PALETTE[id % len(COLOR_PALETTE)] if id >= 0 else (0, 0, 0)

        if show_all_detections or id >= 0:
            cv.rectangle(frame, (left, top), (right, bottom), box_color, thickness=3)

        if id >= 0:
            label = 'ID {}'.format(label) if not isinstance(label, str) else label
            person_class, detect_lines, person_action = person_class_dict.get(id, '')
            label = f'{label} {person_class} {person_action.value}'
            label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 1, 2)
            top = max(top, label_size[1])
            for detect_line in detect_lines:
                detect_line = [int(l) for l in detect_line]
                cv.line(frame, (detect_line[0], detect_line[1]), (detect_line[2], detect_line[3]), (0, 0, 255), thickness=6)
            cv.rectangle(frame, (left, top - label_size[1]), (left + label_size[0], top + base_line),
                         (255, 255, 255), cv.FILLED)
            cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)


def get_target_size(frame_sizes, vis=None, max_window_size=(1920, 1080), stack_frames='vertical', **kwargs):
    if vis is None:
        width = 0
        height = 0
        for size in frame_sizes:
            if width > 0 and height > 0:
                if stack_frames == 'vertical':
                    height += size[1]
                elif stack_frames == 'horizontal':
                    width += size[0]
            else:
                width, height = size
    else:
        height, width = vis.shape[:2]

    if stack_frames == 'vertical':
        target_height = max_window_size[1]
        target_ratio = target_height / height
        target_width = int(width * target_ratio)
    elif stack_frames == 'horizontal':
        target_width = max_window_size[0]
        target_ratio = target_width / width
        target_height = int(height * target_ratio)
    return target_width, target_height


def visualize_multicam_detections(frame_times, frames, all_objects, person_class_dict, fps='', show_all_detections=True,
                                  max_window_size=(1920, 1080), stack_frames='vertical'):
    assert len(frames) == len(all_objects)
    assert stack_frames in ['vertical', 'horizontal']
    vis = None
    for i, (frame_time, frame, objects) in enumerate(zip(frame_times, frames, all_objects)):
        draw_detections(frame, objects, person_class_dict, show_all_detections)
        if vis is not None:
            if stack_frames == 'vertical':
                vis = np.vstack([vis, frame])
            elif stack_frames == 'horizontal':
                vis = np.hstack([vis, frame])
        else:
            vis = frame

    target_width, target_height = get_target_size(frames, vis, max_window_size, stack_frames)

    vis = cv.resize(vis, (target_width, target_height))

    min_frame_time = min(frame_times)
    label_size, base_line = cv.getTextSize(str(min_frame_time), cv.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv.putText(vis, str(min_frame_time), (base_line*2, base_line*3),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 10)
    cv.putText(vis, str(min_frame_time), (base_line*2, base_line*3),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return vis


def plot_timeline(sct_id, last_frame_num, tracks, save_path='', name='', show_online=False):
    def find_max_id():
        max_id = 0
        for track in tracks:
            if isinstance(track, dict):
                track_id = track['id']
            else:
                track_id = track.id
            if track_id > max_id:
                max_id = track_id
        return max_id

    if not show_online and not len(save_path):
        return
    plot_name = '{}#{}'.format(name, sct_id)
    plt.figure(plot_name, figsize=(24, 13.5))
    last_id = find_max_id()
    xy = np.full((last_id + 1, last_frame_num + 1), -1, dtype='int32')
    x = np.arange(last_frame_num + 1, dtype='int32')
    y = np.arange(last_id + 1, dtype='int32')

    plt.xticks(x)
    plt.yticks(y)
    plt.xlabel('Frame')
    plt.ylabel('Identity')

    colors = []
    for track in tracks:
        if isinstance(track, dict):
            frame_ids = track['timestamps']
            track_id = track['id']
        else:
            frame_ids = track.timestamps
            track_id = track.id
        if frame_ids[-1] > last_frame_num:
            frame_ids = [timestamp for timestamp in frame_ids if timestamp < last_frame_num]
        xy[track_id][frame_ids] = track_id
        xx = np.where(xy[track_id] == -1, np.nan, x)
        if track_id >= 0:
            color = COLOR_PALETTE[track_id % len(COLOR_PALETTE)] if track_id >= 0 else (0, 0, 0)
            color = [x / 255 for x in color]
        else:
            color = (0, 0, 0)
        colors.append(tuple(color[::-1]))
        plt.plot(xx, xy[track_id], marker=".", color=colors[-1], label='ID#{}'.format(track_id))
    if save_path:
        file_name = os.path.join(save_path, 'timeline_{}.jpg'.format(plot_name))
        plt.savefig(file_name, bbox_inches='tight')
    if show_online:
        plt.draw()
        plt.pause(0.01)
