#!/usr/bin/env python3

"""
 Copyright (c) 2019-2020 Intel Corporation
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

import argparse
import datetime
import json
import logging as log
import os
import queue
import random
import subprocess
import sys
import time
from collections import defaultdict
from os.path import splitext
from pathlib import Path
from threading import Lock, Thread

import cv2 as cv
from openvino.inference_engine import \
    IECore  # pylint: disable=import-error,E0611

from .mc_tracker.mct import MultiCameraTracker
from .utils.analyzer import save_embeddings
from .utils.classify_persons import classify_persons_per_frame
from .utils.misc import (AverageEstimator, check_pressed_keys, read_py_config,
                         set_log_config)
from .utils.network_wrappers import (DetectionsFromFileReader, Detector,
                                     MaskRCNN, VectorCNN)
from .utils.network_wrappers_yolo import YOLOV4, YOLOV4Tiny
from .utils.video import MulticamCapture, NormalizerCLAHE
from .utils.visualization import get_target_size, visualize_multicam_detections

sys.path.append(os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), 'common'))

# Please import monitors here.
import monitors

set_log_config()

OUTPUT_VIDEO_SIZE_LIMIT = 1000 * 1024 * 1024 # 1GB
MAX_GET_FRAME_TIMES = 50

threading_lock = Lock()
def check_detectors(args):
    detectors = {
        '--m_detector': args.m_detector,
        '--m_segmentation': args.m_segmentation,
        '--detections': args.detections
    }
    non_empty_detectors = [(det, value)
                           for det, value in detectors.items() if value]
    det_number = len(non_empty_detectors)
    if det_number == 0:
        log.error('No detector specified, please specify one of the following parameters: '
                  '\'--m_detector\', \'--m_segmentation\' or \'--detections\'')
    elif det_number > 1:
        det_string = ''.join('\n\t{}={}'.format(
            det[0], det[1]) for det in non_empty_detectors)
        log.error('Only one detector expected but got {}, please specify one of them:{}'
                  .format(len(non_empty_detectors), det_string))
    return det_number


def update_detections(output, detections, frame_number):
    for i, detection in enumerate(detections):
        entry = {'frame_id': frame_number, 'scores': [], 'boxes': []}
        for det in detection:
            entry['boxes'].append(det[0])
            entry['scores'].append(float(det[1]))
        output[i].append(entry)


def save_json_file(save_path, data, description=''):
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_path, 'w') as outfile:
        json.dump(data, outfile)
    if description:
        log.info('{} saved to {}'.format(description, save_path))


class FramesThreadBody:
    def __init__(self, capture, max_queue_length=2):
        self.process = True
        self.frames_queue = queue.Queue()
        self.seconds_queue = queue.Queue()
        self.capture = capture
        self.max_queue_length = max_queue_length
        self.retry_times = 0

    def __call__(self):
        while self.process:
            if self.frames_queue.qsize() > self.max_queue_length:
                time.sleep(0.1)
                continue
            has_frames, frames, seconds = self.capture.get_frames()
            if not has_frames and self.frames_queue.empty():
                self.retry_times +=1
                if self.retry_times >= MAX_GET_FRAME_TIMES:
                    self.process = False
                    log.warn(f'No frames for {self.retry_times} times, exit.')
                    break
            if has_frames:
                self.retry_times = 0
                self.frames_queue.put(frames)
                self.seconds_queue.put(seconds)


def run(params, config, capture, detector, reid, classify_person_flow=None):
    win_name = 'Multi camera tracking'
    frame_number = 0
    avg_latency = AverageEstimator()
    output_detections = [[] for _ in range(capture.get_num_sources())]
    key = -1

    if config['normalizer_config']['enabled']:
        capture.add_transform(
            NormalizerCLAHE(
                config['normalizer_config']['clip_limit'],
                config['normalizer_config']['tile_size'],
            )
        )

    tracker = MultiCameraTracker(capture.get_num_sources(), reid, config['sct_config'], **config['mct_config'],
                                 visual_analyze=config['analyzer'])

    thread_body = FramesThreadBody(
        capture, max_queue_length=len(capture.captures) * 2)
    frames_thread = Thread(target=thread_body)
    frames_thread.start()

    rtsp_mode = params.output_video.startswith('rtsp://')
    if len(params.output_video):
        frame_size, source_fps = capture.get_source_parameters()
        if params.fps:
            source_fps = [params.fps]
        target_width, target_height = get_target_size(
            frame_size, None, **config['visualization_config'])
        video_output_size = (target_width, target_height)
        fourcc = cv.VideoWriter_fourcc(*'XVID')

        if rtsp_mode:
            # https://gist.github.com/takidog/2c981c34d5d5b41c0d712f8ef4ac60d3
            # E.g. rtsp://localhost:8554/output
            # Use "-c copy" would increase video quality but slowdown process
            command = ['ffmpeg',
                       '-i', '-',
                       '-f', 'rtsp',
                       params.output_video]

            output_video = subprocess.Popen(command, stdin=subprocess.PIPE)
        else:
            output_video = cv.VideoWriter(
                params.output_video, fourcc, min(source_fps), video_output_size)
    else:
        output_video = None

    prev_frames = thread_body.frames_queue.get()
    detector.run_async(prev_frames, frame_number)
    presenter = monitors.Presenter(params.utilization_monitors, 0)

    start_time = datetime.datetime.now()
    action_to_person_ids = defaultdict(set)

    while thread_body.process:
        try:
            tick = datetime.datetime.now()

            if not params.no_show:
                key = check_pressed_keys(key)
                if key == 27:
                    break
                presenter.handleKey(key)
            start = time.perf_counter()
            try:
                frames = thread_body.frames_queue.get_nowait()
                seconds = thread_body.seconds_queue.get_nowait()
            except queue.Empty:
                frames = None
                seconds = None

            if frames is None:
                continue

            all_detections = detector.wait_and_grab()
            if params.save_detections:
                update_detections(output_detections, all_detections, frame_number)
            frame_number += 1

            frame_times = [start_time + datetime.timedelta(0, s) for s in seconds]

            detector.run_async(frames, frame_number)

            all_masks = [[] for _ in range(len(all_detections))]
            for i, detections in enumerate(all_detections):
                all_detections[i] = [det[0] for det in detections]
                all_masks[i] = [det[2] for det in detections if len(det) == 3]

            tracker.process(prev_frames, all_detections, all_masks)
            tracked_objects = tracker.get_tracked_objects()

            latency = max(time.perf_counter() - start, sys.float_info.epsilon)
            avg_latency.update(latency)
            fps = round(1. / latency, 1)

            person_class_dict = {}
            if classify_person_flow:
                # Crop persons to classify before drawing
                person_class_dict = classify_persons_per_frame(
                    frame_times, prev_frames, tracked_objects, classify_person_flow, **config['visualization_config'])
                for person_id, (person_class, detect_lines, person_action) in person_class_dict.items():
                    if person_action:
                        action_to_person_ids[person_action].add(person_id)

            vis = visualize_multicam_detections(
                frame_times, prev_frames, tracked_objects, action_to_person_ids, person_class_dict, fps, **config['visualization_config'])
            presenter.drawGraphs(vis)
            if not params.no_show:
                cv.imshow(win_name, vis)

            if output_video:
                if rtsp_mode:
                    ret, frame = cv.imencode('.jpg', vis)
                    if ret:
                        output_video.stdin.write(frame.tobytes())
                else:
                    output_video.write(cv.resize(vis, video_output_size))
                    output_video_file = Path(params.output_video)
                    if output_video_file.stat().st_size > OUTPUT_VIDEO_SIZE_LIMIT:
                        output_video_file.unlink()
                        output_video = cv.VideoWriter(
                            params.output_video, fourcc, min(source_fps), video_output_size)

            if params.output_image:
                # https://blog.gtwang.org/programming/python-threading-multithreaded-programming-tutorial/
                Thread(target = write_output_image, args = (params.output_image, vis,)).start()
            # print('\rProcessing frame: {}, fps = {} (avg_fps = {:.3})'.format(
            #                     frame_number, fps, 1. / avg_latency.get()), end="")
            prev_frames, frames = frames, prev_frames

            tock = datetime.datetime.now()  
            diff = tock - tick 

            # https://stackoverflow.com/questions/5419389/how-to-overwrite-the-previous-print-to-stdout-in-python
            print(frame_times, f'takes {diff.total_seconds()}s', end='\r')
        except Exception as e:
            thread_body.process = False
            raise e
    print(presenter.reportMeans())
    print('')

    thread_body.process = False
    frames_thread.join()

    if len(params.history_file):
        save_json_file(params.history_file, tracker.get_all_tracks_history(
        ), description='History file')
    if len(params.save_detections):
        save_json_file(params.save_detections,
                       output_detections, description='Detections')

    if len(config['embeddings']['save_path']):
        save_embeddings(tracker.scts, **config['embeddings'])

def write_output_image(output_image, vis):
    file_path, ext = splitext(output_image)
    tmp_file_name = f'{file_path}.tmp{ext}'
    threading_lock.acquire()
    cv.imwrite(tmp_file_name, vis, [int(cv.IMWRITE_JPEG_QUALITY), 80])
    os.rename(tmp_file_name, output_image)
    threading_lock.release()


def main(classify_person_flow=None, inputs=None, output_video=None, output_image=None):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    """Prepares data for the object tracking demo"""
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@', description='Multi camera multi object \
                                                  tracking live demo script')
    parser.add_argument('-i', type=str, nargs='+', help='Input sources (indexes \
                            of cameras or paths to video files)', default=inputs, required=False)
    parser.add_argument('--config', type=str, default=os.path.join(current_dir, 'configs/person.py'), required=False,
                        help='Configuration file')

    parser.add_argument('--detections', type=str,
                        help='JSON file with bounding boxes')

    parser.add_argument('-m', '--m_detector', type=str, required=False,
                        help='Path to the object detection model')
    parser.add_argument('--t_detector', type=float, default=0.6,
                        help='Threshold for the object detection model')

    parser.add_argument('--m_segmentation', type=str, required=False,
                        help='Path to the object instance segmentation model')
    parser.add_argument('--t_segmentation', type=float, default=0.6,
                        help='Threshold for object instance segmentation model')

    parser.add_argument('--m_reid', type=str, required=False,
                        help='Path to the object re-identification model')

    parser.add_argument('--output_video', type=str, default=output_video, required=False,
                        help='Optional. Path to output video')
    parser.add_argument('--output_image', type=str, default=output_image, required=False,
                        help='Optional. Path to output image')

    parser.add_argument('--history_file', type=str, default='', required=False,
                        help='Optional. Path to file in JSON format to save results of the demo')
    parser.add_argument('--save_detections', type=str, default='', required=False,
                        help='Optional. Path to file in JSON format to save bounding boxes')
    parser.add_argument(
        "--no_show", help="Optional. Don't show output", action='store_true')

    parser.add_argument('-d', '--device', type=str, default='CPU')
    parser.add_argument('-l', '--cpu_extension',
                        help='MKLDNN (CPU)-targeted custom layers.Absolute \
                              path to a shared library with the kernels impl.',
                             type=str, default=None)
    parser.add_argument('-u', '--utilization_monitors', default='', type=str,
                        help='Optional. List of monitors to show initially.')
    parser.add_argument('--fps', type=float, required=True)
    parser.add_argument("--seek_mode", help="", action='store_true')
    args, _ = parser.parse_known_args()
    if check_detectors(args) != 1:
        sys.exit(1)

    if len(args.config):
        log.info('Reading configuration file {}'.format(args.config))
        config = read_py_config(args.config)
    else:
        log.error(
            'No configuration file specified. Please specify parameter \'--config\'')
        sys.exit(1)

    random.seed(config['random_seed'])
    capture = MulticamCapture(args.i, args.seek_mode, args.fps)

    log.info("Creating Inference Engine")
    ie = IECore()

    if args.detections:
        object_detector = DetectionsFromFileReader(
            args.detections, args.t_detector)
    elif args.m_segmentation:
        object_detector = MaskRCNN(ie, args.m_segmentation,
                                   config['obj_segm']['trg_classes'],
                                   args.t_segmentation,
                                   args.device, args.cpu_extension,
                                   capture.get_num_sources())
    else:
        if 'yolov4' in args.m_detector:
            # Person class index is 0
            trg_classes = [0]
            if 'tiny' in args.m_detector:
                out_blob = 'ALL'
                object_detector = YOLOV4Tiny(ie, args.m_detector,
                                trg_classes,
                                args.t_detector,
                                args.device, args.cpu_extension,
                                capture.get_num_sources(), out_blob=out_blob)
            else:
                out_blob='output'
                object_detector = YOLOV4(ie, args.m_detector,
                                trg_classes,
                                args.t_detector,
                                args.device, args.cpu_extension,
                                capture.get_num_sources(), out_blob=out_blob)
        else:
            object_detector = Detector(ie, args.m_detector,
                                    config['obj_det']['trg_classes'],
                                    args.t_detector,
                                    args.device, args.cpu_extension,
                                    capture.get_num_sources())

    if args.m_reid:
        object_recognizer = VectorCNN(
            ie, args.m_reid, args.device, args.cpu_extension)
    else:
        object_recognizer = None

    run(args, config, capture, object_detector,
        object_recognizer, classify_person_flow)
    log.info('Finished successfully')


if __name__ == '__main__':
    main()
