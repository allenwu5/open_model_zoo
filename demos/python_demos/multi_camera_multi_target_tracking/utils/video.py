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

import logging as log
import os

import cv2 as cv
import ffmpeg
import numpy as np


class MulticamCapture:
    def __init__(self, sources, seek_mode, specific_fps):
        assert sources
        self.sources = sources
        self.captures = []
        self.transforms = []
        self.seek_time = 0
        self.seek_mode = seek_mode
        self.specific_fps = specific_fps

        try:
            self.sources = [int(src) for src in sources]
            mode = 'cam'
        except ValueError:
            mode = 'video'

        if mode == 'cam':
            for id in sources:
                log.info('Connection  cam {}'.format(id))
                cap = cv.VideoCapture(id)
                cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv.CAP_PROP_FPS, 30)
                cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
                assert cap.isOpened()
                self.captures.append(cap)
        else:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            for video_path in sources:
                log.info('Opening file {}'.format(video_path))
                cap = cv.VideoCapture(video_path)
                assert cap.isOpened()
                self.captures.append(cap)

    def add_transform(self, t):
        self.transforms.append(t)

    def get_frames(self):
        frames = []
        seconds = []

        if not self.seek_mode:
            for capture in self.captures:
                has_frame, frame = capture.read()
                if has_frame:
                    frame_time = capture.get(cv.CAP_PROP_POS_MSEC)
                    if frame_time >= self.seek_time * 1000:
                        for t in self.transforms:
                            frame = t(frame)
                        frames.append(frame)
                        seconds.append(self.seek_time)
                        self.seek_time += 1/self.specific_fps
        else:
            for source in self.sources:
                out, _ = (ffmpeg
                          .input(source, ss=self.seek_time)
                          .output('pipe:', vframes=1, format='image2', vcodec='mjpeg')
                          .run(capture_stdout=True, quiet=True)
                          )
                if out:
                    frame = cv.imdecode(np.frombuffer(out, np.uint8), -1)
                    frames.append(frame)
                    seconds.append(self.seek_time)
                    self.seek_time += 1/self.specific_fps

        return len(frames) == len(self.captures), frames, seconds

    def get_num_sources(self):
        return len(self.captures)

    def get_source_parameters(self):
        frame_size = []
        fps = []
        for cap in self.captures:
            frame_size.append((int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
                               int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
            fps.append(int(cap.get(cv.CAP_PROP_FPS)))
        return frame_size, fps


class NormalizerCLAHE:
    def __init__(self, clip_limit=.5, tile_size=16):
        self.clahe = cv.createCLAHE(clipLimit=clip_limit,
                                    tileGridSize=(tile_size, tile_size))

    def __call__(self, frame):
        for i in range(frame.shape[2]):
            frame[:, :, i] = self.clahe.apply(frame[:, :, i])
        return frame
