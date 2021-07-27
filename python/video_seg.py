import matplotlib.pyplot as plt
import argparse
import os
import cv2
from itertools import cycle
import numpy as np
import time
import torch
from torch.utils.data import IterableDataset

from monai.networks.utils import eval_mode
from monai.transforms import DivisiblePad, Compose, AsDiscrete, Activations, ScaleIntensity, ToTensor, AsChannelFirst, Lambda

class VideoDataset(IterableDataset):
    def __init__(self, video_source, divisible_pad_factor, device, max_num_loops):
        self.device = device
        self.transforms = Compose([
            AsChannelFirst(),
            Lambda(lambda x: x[::-1,...]),
            DivisiblePad(divisible_pad_factor),
            ScaleIntensity(),
            ToTensor(),
        ])
        self.max_num_loops = max_num_loops
        self.cap = self.open_video(video_source)

    @staticmethod
    def open_video(video_source):
        if isinstance(video_source, str) and not os.path.isfile(video_source):
            raise RuntimeError("Video file does not exist: " + video_source)
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError("Failed to open video: " + video_source)
        return cap

    def get_next_frame(self, frame):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {frame}")
        return self.preprocess_frame(frame)

    def preprocess_frame(self, frame):
        """Perform transforms, change to float, move to device, add batch dimension."""
        return self.transforms(frame).float().to(self.device)[None]

    def __iter__(self):
        for i in range(self.max_num_loops):
            frame = self.get_next_frame(i)
            yield frame

class VideoFileDataset(VideoDataset):
    def __init__(self, video_source, divisible_pad_factor, device, max_num_loops):
        super().__init__(video_source, divisible_pad_factor, device, max_num_loops)
        num_frames = self.get_num_frames(self.cap)
        if self.max_num_loops is None or num_frames < self.max_num_loops:
            self.max_num_loops = num_frames

    @staticmethod
    def get_num_frames(cap):
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if num_frames == 0:
            raise RuntimeError("0 frames found")
        print(f"num frames:    {num_frames}")
        return num_frames

class LoopDataset(VideoFileDataset):
    def __init__(self, video_source, divisible_pad_factor, device, max_num_loops):
        super().__init__(video_source, divisible_pad_factor, device, max_num_loops)

        self.frames = [self.get_next_frame(i) for i in range(5)]
        self.frame_iter = cycle(self.frames)

    def __iter__(self):
        for _ in range(self.max_num_loops):
            frame = next(self.frame_iter)
            yield frame


class CameraDataset(VideoDataset):
    def __init__(self, stream_device, divisible_pad_factor, device, max_num_loops):
        super().__init__(stream_device, divisible_pad_factor, device, max_num_loops)

    @staticmethod
    def get_possible_devices():
        index = 0
        arr = []
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:
                break
            else:
                arr.append(index)
            cap.release()
            index += 1
        return arr

def imshows(ims, names, im_show_handles):
    # create new plots
    if len(im_show_handles) == 0:
        fig, axes = plt.subplots(1, len(ims))
        im_show_handles = []
        for im, title, ax in zip(ims, names, axes):
            ax.set_title(f"{title}\n{im.shape}")
            im_show_handles.append(ax.imshow(im))
            ax.axis("off")
            fig.colorbar(im_show_handles[-1], ax=ax)

    # if already created the imshows, just update them
    else:
        for im, handle in zip(ims, im_show_handles):
            handle.set_data(im)

    plt.pause(0.001)
    return im_show_handles

def imshow(im, output, im_show_handles):
    def as_numpy(im):
        im = im[0].detach().cpu().numpy()
        return im[0] if im.shape[0] == 1 else np.moveaxis(im, 0, -1)

    def normalise_to(a, _min, _max):
        return np.interp(a, (a.min(), a.max()), (_min, _max))

    im_np = normalise_to(as_numpy(im), 0, 1)
    out_np = normalise_to(as_numpy(output), 0, 1)

    # need to flip rgb because opencv is bgr
    im_show_handles = imshows((im_np, out_np), ("im", "label"), im_show_handles)

    return im_show_handles

def main(video_source, model_path, device, divisible_pad, mode, max_num_loops, show):

    device = torch.device(device)

    model = torch.jit.load(model_path)
    model.to(device)
    with eval_mode(model):

        if mode == "video":
            Dataset = VideoFileDataset
        elif mode == "pre_process":
            Dataset = LoopDataset
        else:
            Dataset = CameraDataset
            video_source = int(video_source)
        dataset = Dataset(video_source, divisible_pad, device, max_num_loops)

        if device == torch.device("cuda"):
            torch.cuda.synchronize()
        t_start = time.time()

        post_trans = Compose([
            Activations(sigmoid=True),
            AsDiscrete(threshold_values=True),
        ])

        for i, im in enumerate(dataset):
            output = model(im)
            output = post_trans(output)

            if i == 0:
                print("input shape:  ", im.shape)
                print("output shape: ", output.shape)

            if show:
                if i == 0:
                    im_show_handles = []
                im_show_handles = imshow(im, output, im_show_handles)

        if device == torch.device("cuda"):
            torch.cuda.synchronize()
        t_end = time.time()
        fps = (i+1) / (t_end-t_start)
        print(f"processed {i+1} frames at an average of {fps:.5} FPS.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_source",
        help="Path or device index (in camera mode) of video to segment.",
        required=True)
    parser.add_argument("--model_path",
        help="Path of model to use for segmentation",
        required=True)
    parser.add_argument("--device",
        help="Device for segmentation",
        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--divisible_pad",
        help="Factor for DivisiblePad",
        default=16,
        type=int)
    parser.add_argument("--mode",
        help="Segment video from file [video], segment from file looping only first 5 frames (removes OpenCV bottleneck) [pre_process], or use camera (e.g., webcam) [camera]",
        choices=["video", "pre_process", "camera"],
        required=True)
    parser.add_argument("--max_num_loops",
        help="Maximum number of loops to perform. Required for --mode \"camera\" (but can also be used with other modes, where default is number of frames in video).",
        type=int,
        default=None)
    parser.add_argument("--list_possible_cameras",
        help="List possible cameras and then exit.",
        action="store_true")
    parser.add_argument("--show",
        help="show input and output videos",
        action="store_true")
    args = parser.parse_args()

    if args.list_possible_cameras:
        CameraDataset.get_possible_devices()
        exit(0)

    if args.mode == "camera" and args.max_num_loops is None:
        raise RuntimeError("If using in camera mode, --max_num_loops is required")

    print(f"video source:  {args.video_source}")
    print(f"model path:    {args.model_path}")
    print(f"device:        {args.device}")
    print(f"divisible pad: {args.divisible_pad}")
    print(f"mode:          {args.mode}")
    print(f"max_num_loops: {args.max_num_loops}")
    print(f"show:          {args.show}")
    main(args.video_source, args.model_path, args.device, args.divisible_pad, args.mode, args.max_num_loops, args.show)
