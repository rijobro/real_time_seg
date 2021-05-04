import cv2
from itertools import cycle
import numpy as np
import sys
import time
import torch
from torch.utils.data import IterableDataset

from monai.networks.nets import UNet
from monai.networks.utils import eval_mode
from monai.transforms import DivisiblePad


class VideoDataset(IterableDataset):
    def __init__(self, video_fname, divisible_pad_factor, device):
        self.video_path = video_fname
        self.device = device
        self.cap = cv2.VideoCapture(video_fname)
        self.padder = DivisiblePad(divisible_pad_factor)
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self):
        for i in range(self.num_frames):
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = torch.Tensor(self.padder(np.moveaxis(frame, -1, 0))[None])
            frame = frame.to(self.device)
            yield frame


class LoopDataset(VideoDataset):
    def __init__(self, video_fname, divisible_pad_factor, device):
        super().__init__(video_fname, divisible_pad_factor, device)

        self.frames = []
        for _ in range(5):
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError
            frame = self.padder(np.moveaxis(frame, -1, 0))[None]
            self.frames.append(torch.Tensor(frame).to(device))
            self.frame_iter = cycle(self.frames)

    def __iter__(self):
        for _ in range(self.num_frames):
            yield next(self.frame_iter)



def main(video_fname, model_path, device, pre_process_5_frames):

    print(f"\nvideo_fname: {video_fname}")
    print(f"model_path: {model_path}")

    device = torch.device(device)
    print(f"\ndevice: {device}\n")

    model = torch.jit.load(model_path)
    model.to(device)
    with eval_mode(model):

        divisible_pad_factor = 16
        Dataset = VideoDataset if not pre_process_5_frames else LoopDataset
        dataset = Dataset(video_fname, divisible_pad_factor, device)

        if device==torch.device("cuda"):
            torch.cuda.synchronize()
        t_start = time.time()

        for i, im in enumerate(dataset):
            if i==0:
                print("tensor shape: ", im.shape)
            output = model(im)
            if i==0:
                print("output shape: ", output.shape)

        if device==torch.device("cuda"):
            torch.cuda.synchronize()
        t_end = time.time()
        fps = (i+1) / (t_end-t_start)
        print(f"processed {i+1} frames at an average of {fps:.5} FPS.")


if __name__ == "__main__":
    vid_fname = sys.argv[1]
    model_path = sys.argv[2]
    device = sys.argv[3] if len(sys.argv) > 3 else "cuda" if torch.cuda.is_available() else "cpu"
    pre_process_5_frames = bool(int(sys.argv[4])) if len(sys.argv) > 4 else False
    main(vid_fname, model_path, device, pre_process_5_frames)
