import argparse
import os
import platform
import sys
from pathlib import Path
import cv2
import torch

from ultralytics import YOLO

import pyautogui
from PIL import ImageGrab
import win32gui, win32con, win32com.client
import numpy as np
import time


def cilck_init():
    hwnd = win32gui.FindWindow(None, '植物大战僵尸中文版')
    print(hwnd)
    shell = win32com.client.Dispatch("WScript.Shell")
    shell.SendKeys('%')
    win32gui.SetForegroundWindow(hwnd)
    window_x, window_y, right, bottom = win32gui.GetWindowRect(hwnd)
    box = (window_x, window_y, right, bottom)
    print(box)
    return box


def run(weights='runs/detect/train/weights/best.pt', source='self_data/pvz', imgsz=640, conf_thres=0.25,
        iou_thres=0.45):
    # Load model
    model = YOLO(weights)

    # Initialize click function
    box = cilck_init()

    num_pic = 1
    while num_pic:
        # Grab screenshot
        background_bgr = np.array(ImageGrab.grab(box))
        background = background_bgr[:, :, [2, 1, 0]]  # Convert BGR to RGB
        img_path = 'datasets/data/pvz/test.jpg'
        cv2.imwrite(img_path, background)

        # Perform inference
        results = model.predict(img_path, imgsz=imgsz, conf=conf_thres, iou=iou_thres)

        # Process results
        for result in results:
            for det in result.boxes.data:
                xyxy = det[:4].cpu().numpy().astype(int)
                conf = det[4].cpu().numpy()
                cls = int(det[5].cpu().numpy())
                print(xyxy, conf, cls)

                # Use pyautogui to click on detected coordinates
                pyautogui.click(box[0] + xyxy[0], box[1] + xyxy[1] + 20)

                # Additional processing or saving results can be added here

        num_pic += 1
        #time.sleep(1)  # Add a delay to avoid excessive clicking


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/detect/train/weights/best.pt', help='model path')
    parser.add_argument('--source', type=str, default='self_data/pvz', help='source')
    parser.add_argument('--imgsz', type=int, default=640, help='image size')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold')
    opt = parser.parse_args()

    run(opt.weights, opt.source, opt.imgsz, opt.conf_thres, opt.iou_thres)


if __name__ == "__main__":
    main()
