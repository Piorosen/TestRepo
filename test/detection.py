
import json
import cv2
from utils.utils import get_yolo_boxes, makedirs
from utils.bbox import draw_boxes
from keras.models import load_model
import numpy as np

def predict(image):
    config_path = "./config_license_plates.json"
    net_h, net_w = 416, 416  # a multiple of 32, the smaller the faster
    obj_thresh, nms_thresh = 0.5, 0.45
    infer_model = load_model("./license_plate.h5")

    anchors = [15,6, 18,8, 22,9, 27,11, 32,13, 41,17, 54,21, 66,27, 82,33]
    boxes = get_yolo_boxes(infer_model, [image], net_h, net_w, anchors, obj_thresh, nms_thresh)[0]
    return boxes

def draw_box(image, boxes, obj_thresh):
    draw_boxes(image, boxes, ["lp"], obj_thresh)
    return image

