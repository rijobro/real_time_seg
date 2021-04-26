import cv2
from itertools import cycle
import numpy as np
import sys
import time
import torch
import torch.autograd.profiler as profiler
from torch.utils.data import IterableDataset

from monai.networks.nets import UNet
from monai.networks.utils import eval_mode
from monai.transforms import DivisiblePad

try:
    from cv2 import cuda_FarnebackOpticalFlow
    HAS_NVIDIA_OPENCV = True
except:
    HAS_NVIDIA_OPENCV = False

class LoopDataset(IterableDataset):
    def __init__(self, video_fname, divisible_pad_factor, device, max_iters, pre_process_and_loop_5_frames=False):
        self.video_path = video_fname
        self.device = device
        self.cap = cv2.VideoCapture(video_fname)
        self.padder = DivisiblePad(divisible_pad_factor)
        self.max_iters = max_iters
        self.pre_process_and_loop_5_frames = pre_process_and_loop_5_frames

        if pre_process_and_loop_5_frames:
            self.frames = []
            for _ in range(5):
                ret, frame = self.cap.read()
                if not ret:
                    raise RuntimeError
                frame = self.padder(np.moveaxis(frame, -1, 0))[None]
                self.frames.append(torch.Tensor(frame).to(device))

    def __iter__(self):
        gpu_frame = None

        for i in range(self.max_iters):

            if self.pre_process_and_loop_5_frames:
                if i == 0:
                    frame_iter = cycle(self.frames)
                gpu_frame = next(frame_iter)
            else:
                ret, frame = self.cap.read()
                if not ret:
                    break
                # if not HAS_NVIDIA_OPENCV:
                frame = torch.Tensor(self.padder(np.moveaxis(frame, -1, 0))[None])
                gpu_frame = frame.to(self.device)
                # else:
                    # gpu_frame.upload(frame)
            yield gpu_frame


class MyUNet(UNet):
    def create_jit_model(self, x, method):
        self.model = method(self.model, x)



def main(video_fname, model_path, use_jit, pre_process_5_frames):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MyUNet(
        dimensions=2,
        in_channels=3,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    with eval_mode(model):

        divisible_pad_factor = 16
        max_iters = 100
        dataset = LoopDataset(video_fname, divisible_pad_factor,
                              device, max_iters, pre_process_5_frames)
        #with profiler.profile(record_shapes=True, profile_memory=True, use_cuda=True) as prof:
        for i, im in enumerate(dataset):
            if i==0:
                if use_jit:
                    model.create_jit_model(im, torch.jit.trace)
                    # print("model_trace code:")
                    # print(model.model.code)
                    # exit(0)
                torch.cuda.synchronize()
                t_start = time.time()
            print(im.shape)
            exit(0)
            _ = model(im)[0]
        torch.cuda.synchronize()
        t_end = time.time()
        print(f"processed {i+1} frames at an average of {(i+1) / (t_end-t_start)} FPS.")

        #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


if __name__ == "__main__":
    vid_fname = sys.argv[1]
    model_path = sys.argv[2]
    use_jit = bool(int(sys.argv[3]))
    pre_process_5_frames = bool(int(sys.argv[4]))
    main(vid_fname, model_path, use_jit, pre_process_5_frames)
