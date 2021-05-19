# Real time segmentation (and a bit of optical flow)

A few small scripts that take a torch.jit model and perform segmentation/optical flow.

# Data
The example [video](example_data/videos/d1_im.mp4) and its corresponding [ground truth label](example_data/videos/d1_seg.mp4) were taken from the 2017 MICCAI EndoVis Grand Challenge. They are at 2 FPS and consist of 202 frames.

The example model is MONAI's UNet, the parameters of which are given below. A quick training was performed with the individual frames of all the vidoes in the 2017 EndoVis challenge. The training was quick and the results from the model aren't meant to be great, they're just there to give a rough idea. The training script can be found [here](https://github.com/rijobro/Tutorials/blob/video_seg/modules/video_seg.ipynb).

```python
UNet(
    dimensions=2,
    in_channels=3,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)
```

# Segmentation
## Python 

From the base level, use the following to segment a video file:

```bash
python python/video_seg.py \
	--model_path example_data/models/model_jit.pt \
	--mode video --video_source example_data/videos/d1_im.mp4
```

To segment a video file and display the live results:

```bash
python python/video_seg.py \
	--model_path example_data/models/model_jit.pt \
	--mode video --video_source example_data/videos/d1_im.mp4 --show
```

To use the webcam 0 and display the live results:

```bash
python python/video_seg.py \
	--model_path example_data/models/model_jit.pt \
	--mode camera --video_source 0 --max_num_loops 100 --show
```

## C++
You'll need `libtorch` and a C++ implementation of `OpenCV`. The Docker image I use with these installed (plus some other stuff) is [here](https://hub.docker.com/repository/docker/rijobro/rb-monai) (source is [here](https://github.com/rijobro/dgxscripts/blob/main/Dockerfile)).

```bash
git clone https://github.com/rijobro/real_time_seg.git
mkdir real_time_seg_build
cd real_time_seg_build
cmake ../real_time_seg -G Ninja
ninja
device=cuda
#device=cpu
./real_time_seg example_data/videos/d1_im.mp4 example_data/models/model_jit.pt $device
```



# Optical flow
I was interested in the NVIDIA optical flow implementations (see [here](https://developer.nvidia.com/blog/opencv-optical-flow-algorithms-with-nvidia-turing-gpus/)). The implementation should work, but I haven't touched it in a while.
