from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
import json

import cv2
import torch
import numpy as np
from glob import glob

from pysot.tracker import TrackerFactory
from pysot.core.config import cfg

torch.set_num_threads(1)


def parse_args():
    parser = argparse.ArgumentParser(description='tracking demo')
    parser.add_argument('--tracker_type', type=str, help='tracker type')
    parser.add_argument('--config_file', type=str, help='config file')
    parser.add_argument('--model_path', type=str, help='model path')
    parser.add_argument('--video_path', default='', type=str, help='videos or image files')
    args = parser.parse_args()
    return args


def load_video_config(video_path):
    """从与视频同名的json文件加载配置"""
    if not video_path:
        return None

    config_path = os.path.splitext(video_path)[0] + '.json'
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
    return None


def save_video_config(video_path, init_rect, start_time=0, scale_size=None):
    """保存配置到与视频同名的json文件"""
    if not video_path:
        return

    config_path = os.path.splitext(video_path)[0] + '.json'
    config = {
        'init_rect': list(init_rect) if init_rect is not None else None,
        'start_time': start_time,
        'scale_size': scale_size
    }

    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Config saved to {config_path}")
    except Exception as e:
        print(f"Warning: Failed to save config to {config_path}: {e}")


class VideoCapture(object):
    def __init__(self, video_path):
        self.cap = None
        self.images = None

        if not video_path:
            cap = cv2.VideoCapture(0)
            for i in range(5):
                cap.read()
            self.cap = cap
        elif video_path.endswith('avi') or video_path.endswith('mp4'):
            self.cap = cv2.VideoCapture(video_path)
        else:
            images = glob(os.path.join(video_path, '*.jp*'))
            self.images = sorted(images, key=lambda x: int(x.split('/')[-1].split('.')[0]))
            self.img_idx = 0
        
    def get_frame(self):
        frame = None
        if self.cap is not None:
            ret, frame = self.cap.read()
            if not ret: 
                frame = None
        elif self.images is not None:
            if self.img_idx < len(self.images):
                frame = cv2.imread(self.images[self.img_idx])
                self.img_idx += 1
        return frame


def main():
    args = parse_args()

    # build tracker
    tracker = TrackerFactory.create_instance(args.tracker_type, args.config_file, args.model_path)

    # 加载视频配置
    video_path = args.video_path
    video_config = load_video_config(video_path)
    init_rect = None
    start_time = 0
    scale_size = None

    if video_config:
        init_rect = video_config.get('init_rect')
        start_time = video_config.get('start_time', 0)
        scale_size = video_config.get('scale_size')
        print(f"Loaded config: init_rect={init_rect}, start_time={start_time}, scale_size={scale_size}")

    first_frame = True
    paused = False
    step_one = False

    video = VideoCapture(video_path)
    if video_path:
        video_name = video_path.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)

    # 如果有起始时间，跳过前面的帧
    if start_time > 0 and video.cap is not None:
        video.cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        print(f"Starting from {start_time} seconds")

    while True:
        c = cv2.waitKey(10)
        if (c == 27):
            break
        elif (c == 32):
            paused = not paused
        elif (c == 102):
            step_one = True
            paused = True
        if (paused and not first_frame):
            if (not step_one):
                continue
        if not first_frame:
            step_one = False

        frame = video.get_frame()
        if frame is None:
            print("End of video!")
            break

        # 如果有缩放尺寸，缩放图像
        if scale_size is not None and first_frame:
            frame = cv2.resize(frame, tuple(scale_size))

        if first_frame:
            # 检查是否有有效的初始框（存在且尺寸非0）
            use_saved_rect = False
            if init_rect is not None and len(init_rect) == 4:
                if init_rect[2] > 0 and init_rect[3] > 0:
                    use_saved_rect = True
                    print(f"Using saved init_rect: {init_rect}")

            if not use_saved_rect:
                # 交互式选择目标框
                try:
                    init_rect = cv2.selectROI(video_name, frame, False, False)
                except:
                    exit()

            tracker.init(frame, init_rect)
            first_frame = False
            
            bbox = init_rect
            cv2.rectangle(frame, (bbox[0], bbox[1]),
                            (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                            (0, 0, 255), 3)

            # 保存配置（无论是否从文件读取）
            save_video_config(video_path, init_rect, start_time, scale_size)

        else:
            outputs = tracker.track(frame)
            print(f"Tracking outputs: {outputs}")
            if 'polygon' in outputs:
                polygon = np.array(outputs['polygon']).astype(np.int32)
                cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                              True, (0, 255, 0), 3)
                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
            else:
                bbox = list(map(int, outputs['bbox']))
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                              (0, 255, 0), 3)

        cv2.imshow(video_name, frame)


if __name__ == '__main__':
    main()
