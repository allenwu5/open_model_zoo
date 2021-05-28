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


from math import exp as exp

import numpy as np
import torch

from .ie_tools import load_ie_model
from .network_wrappers import DetectorInterface
from .nms import non_max_suppression, scale_coords


class YOLOV4(DetectorInterface):
    """Wrapper class for detector"""

    def __init__(self, ie, model_path, trg_classes, conf=.6,
                 device='CPU', ext_path='', max_num_frames=1, out_blob=None):
        self.net = load_ie_model(ie, model_path, device, None,
                                 ext_path, num_reqs=max_num_frames, out_blob=out_blob)
        self.trg_classes = trg_classes
        self.confidence = conf
        self.expand_ratio = (1., 1.)
        self.max_num_frames = max_num_frames
        self.input_shape = self.net.inputs_info[self.net.input_key].input_data.shape

    def run_async(self, frames, index):
        assert len(frames) <= self.max_num_frames

        frames = [np.true_divide(f, 255.0).astype(np.float32) for f in frames]

        self.shapes = []
        for i in range(len(frames)):
            self.shapes.append(frames[i].shape)
            self.net.forward_async(frames[i])

    def wait_and_grab(self, only_target_class=True):
        all_detections = []
        outputs = self.net.grab_all_async()
        for i, out in enumerate(outputs):
            detections = self.__decode_detections(
                out, self.shapes[i], only_target_class)
            all_detections.append(detections)
        return all_detections

    def get_detections(self, frames):
        """Returns all detections on frames"""
        self.run_async(frames)
        return self.wait_and_grab()

    def __decode_detections(self, out, frame_shape, only_target_class):
        detections = []

        predictions = torch.from_numpy(out)
        predictions = non_max_suppression(
            predictions, self.confidence, iou_thres=0.5, classes=self.trg_classes)

        for prediction in predictions:
            if prediction is None:
                continue
            prediction[:, :4] = scale_coords(
                self.input_shape[2:], prediction[:, :4], frame_shape).round()
            for detection in prediction:
                if only_target_class and detection[-1] not in self.trg_classes:
                    continue

                confidence = detection[-2]
                if confidence < self.confidence:
                    continue

                left = int(max(detection[0], 0))
                top = int(max(detection[1], 0))
                right = int(max(detection[2], 0))
                bottom = int(max(detection[3], 0))
                if self.expand_ratio != (1., 1.):
                    w = (right - left)
                    h = (bottom - top)
                    dw = w * (self.expand_ratio[0] - 1.) / 2
                    dh = h * (self.expand_ratio[1] - 1.) / 2
                    left = max(int(left - dw), 0)
                    right = int(right + dw)
                    top = max(int(top - dh), 0)
                    bottom = int(bottom + dh)

                detections.append(((left, top, right, bottom), confidence))

        if len(detections) > 1:
            detections.sort(key=lambda x: x[1], reverse=True)

        return detections


class YOLOV4Tiny(DetectorInterface):
    """Wrapper class for detector"""

    def __init__(self, ie, model_path, trg_classes, conf=.6,
                 device='CPU', ext_path='', max_num_frames=1, out_blob=None):
        self.net = load_ie_model(ie, model_path, device, None,
                                 ext_path, num_reqs=max_num_frames, out_blob=out_blob)
        self.trg_classes = trg_classes
        self.confidence = conf
        self.expand_ratio = (1., 1.)
        self.max_num_frames = max_num_frames
        self.input_shape = self.net.inputs_info[self.net.input_key].input_data.shape
        self.anchors_dict = {
            13: [81.0, 82.0, 135.0, 169.0, 344.0, 319.0],
            26: [23.0, 27.0, 37.0, 58.0, 81.0, 82.0]
        }

    def run_async(self, frames, index):
        assert len(frames) <= self.max_num_frames

        # frames = [np.true_divide(f, 255.0).astype(np.float32) for f in frames]

        self.shapes = []
        for i in range(len(frames)):
            self.shapes.append(frames[i].shape)
            self.net.forward_async(frames[i])

    def wait_and_grab(self, only_target_class=True):
        all_detections = []
        outputs = self.net.grab_all_async()
        for i, out in enumerate(outputs):
            detections = self.__decode_detections(
                out, self.shapes[i], only_target_class)
            all_detections.append(detections)
        return all_detections

    def get_detections(self, frames):
        """Returns all detections on frames"""
        self.run_async(frames)
        return self.wait_and_grab()

    def __decode_detections(self, outs, frame_shape, only_target_class):
        """
        From https://github.com/TNTWEN/OpenVINO-YOLOV4/blob/master/pythondemo/2021.3/object_detection_demo_yolov3_async.py
        """
        assert only_target_class

        detections = []

        new_outs = None
        bbox_count = 3
        bbox_size = 4 + 1 + 80

        _, _, input_height, input_width = self.input_shape

        for out in outs:
            _, _, row_count, col_count = out.shape
            assert row_count == col_count

            anchors = self.anchors_dict[row_count]

            for row, col, n in np.ndindex(row_count, col_count, bbox_count):
                bbox = out[0, n*bbox_size:(n+1)*bbox_size, row, col]
                x, y, width, height, object_probability = bbox[:5]
                if object_probability < self.confidence:
                    continue

                class_probabilities = bbox[5:]
                # class_id = np.argmax(class_probabilities)
                # Process raw value
                x = (col + x) / col_count
                y = (row + y) / row_count
                # Value for exp is very big number in some cases so following construction is using here
                try:
                    width = exp(width)
                    height = exp(height)
                except OverflowError:
                    continue
                # Depends on topology we need to normalize sizes by feature maps (up to YOLOv3) or by input shape (YOLOv3)
                width = width * anchors[2 * n] / input_width
                height = height * anchors[2 * n + 1] / input_height
                # print(object_probability)
                im_width = frame_shape[1]
                im_height = frame_shape[0]

                x = int(x * im_width)
                y = int(y * im_height)
                width = int(width * im_width)
                height = int(height * im_height)

                bbox[0] = x
                bbox[1] = y
                bbox[2] = width
                bbox[3] = height

                if new_outs is None:
                    new_outs = bbox
                else:
                    new_outs = np.vstack((new_outs, bbox))

        new_outs = np.expand_dims(new_outs, axis=0)
        predictions = torch.from_numpy(new_outs)
        predictions = non_max_suppression(
            predictions, conf_thres=self.confidence, iou_thres=0.5, classes=self.trg_classes)

        for prediction in predictions:
            if prediction is None:
                continue

            for detection in prediction:

                confidence = detection[-2]

                left = int(max(detection[0], 0))
                top = int(max(detection[1], 0))
                right = int(max(detection[2], 0))
                bottom = int(max(detection[3], 0))
                if self.expand_ratio != (1., 1.):
                    w = (right - left)
                    h = (bottom - top)
                    dw = w * (self.expand_ratio[0] - 1.) / 2
                    dh = h * (self.expand_ratio[1] - 1.) / 2
                    left = max(int(left - dw), 0)
                    right = int(right + dw)
                    top = max(int(top - dh), 0)
                    bottom = int(bottom + dh)

                detections.append(((left, top, right, bottom), confidence))

        if len(detections) > 1:
            detections.sort(key=lambda x: x[1], reverse=True)

        return detections
