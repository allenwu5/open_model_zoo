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

import json
import logging as log
import time
from abc import ABC, abstractmethod
from collections import namedtuple

import cv2
import numpy as np

from .ie_tools import load_ie_model
from .segm_postprocess import postprocess

from .network_wrappers import DetectorInterface

import torch
import torchvision


class YOLOV4(DetectorInterface):
    """Wrapper class for detector"""

    def __init__(self, ie, model_path, trg_classes, conf=.6,
                 device='CPU', ext_path='', max_num_frames=1, out_blob=None):
        self.net = load_ie_model(ie, model_path, device, None, ext_path, num_reqs=max_num_frames, out_blob=out_blob)
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
            detections = self.__decode_detections(out, self.shapes[i], only_target_class)
            all_detections.append(detections)
        return all_detections

    def get_detections(self, frames):
        """Returns all detections on frames"""
        self.run_async(frames)
        return self.wait_and_grab()

    def __decode_detections(self, out, frame_shape, only_target_class):
        """Decodes raw SSD output"""
        detections = []

        predictions = torch.from_numpy(out)
        predictions = non_max_suppression(predictions, conf_thres=0.3, iou_thres=0.5, classes=self.trg_classes)
        
        for prediction in predictions:
            if prediction is None:
                continue
            prediction[:, :4] = scale_coords(self.input_shape[2:], prediction[:, :4], frame_shape).round()
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

def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output
def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def convert_frame(frame, img_size):

    # Padded resize
    img = letterbox(frame, new_shape=img_size)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)    
    return img


from math import exp as exp


class YOLOV4Tiny(DetectorInterface):
    """Wrapper class for detector"""

    def __init__(self, ie, model_path, trg_classes, conf=.6,
                 device='CPU', ext_path='', max_num_frames=1, out_blob=None):
        self.net = load_ie_model(ie, model_path, device, None, ext_path, num_reqs=max_num_frames, out_blob=out_blob)
        self.trg_classes = trg_classes
        self.confidence = conf
        self.expand_ratio = (1., 1.)
        self.max_num_frames = max_num_frames
        self.input_shape = self.net.inputs_info[self.net.input_key].input_data.shape

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
            detections = self.__decode_detections(out, self.shapes[i], only_target_class)
            all_detections.append(detections)
        return all_detections

    def get_detections(self, frames):
        """Returns all detections on frames"""
        self.run_async(frames)
        return self.wait_and_grab()

    def __decode_detections(self, outs, frame_shape, only_target_class):
        """Decodes raw SSD output"""
        detections = []

        new_out = None
        n_count = 3
        conf_thres = 0.1
        bbox_size = 4 + 1 + 80
        size_normalizer = (416, 416)

        for out in outs:
            _, _ ,row_count, col_count = out.shape
            assert row_count == 13 or row_count == 26
            assert row_count == col_count
            
            # OpenVINO-YOLOV4/pythondemo/2021.3/object_detection_demo_yolov3_async.py
            # Convert YOLOv4 Tiny (1, 255, 26, 26) to YOLOv4 (1, 2028, 85)
            
            if row_count==26:
                anchors = [23.0, 27.0, 37.0, 58.0, 81.0, 82.0]
            else:
                anchors = [81.0, 82.0, 135.0, 169.0, 344.0, 319.0]
            
            for row, col, n in np.ndindex(row_count, col_count, n_count):
                bbox = out[0, n*bbox_size:(n+1)*bbox_size, row, col]
                x, y, width, height, object_probability = bbox[:5]
                if object_probability < conf_thres:
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
                width = width * anchors[2 * n] / size_normalizer[0]
                height = height * anchors[2 * n + 1] / size_normalizer[1]
                # print(object_probability)
                im_width = 1920
                im_height = 1080


                x = int(x * im_width)
                y = int(y * im_height)
                width = int(width * im_width)
                height = int(height * im_height)

                bbox[0] = x
                bbox[1] = y
                bbox[2] = width
                bbox[3] = height

                if new_out is None:
                    new_out = bbox
                else:
                    new_out = np.vstack((new_out, bbox))


        new_out = np.expand_dims(new_out, axis=0)
        predictions = torch.from_numpy(new_out)
        predictions = non_max_suppression(predictions, conf_thres=conf_thres, iou_thres=0.5, classes=self.trg_classes)
        
        for prediction in predictions:
            if prediction is None:
                continue

            # prediction[:, :4] = scale_coords(self.input_shape[2:], prediction[:, :4], frame_shape).round()

            for detection in prediction:
                if only_target_class and detection[-1] not in self.trg_classes:
                    continue

                confidence = detection[-2]
                if confidence < conf_thres:
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

def scale_bbox(x, y, height, width, im_h, im_w):
    xmin = int((x - width / 2) * im_w)
    ymin = int((y - height / 2) * im_h)
    xmax = int(xmin + width * im_w)
    ymax = int(ymin + height * im_h)
    return xmin, ymin, xmax, ymax
