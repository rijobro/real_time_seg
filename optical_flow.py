import argparse
import cv2
import numpy as np
from tkinter import Tk

# determine if we have NVIDIA CV2
try:
    from cv2 import cuda_FarnebackOpticalFlow
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

def get_window(window_name):
    root = Tk()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)

def get_next_frame(cap, im_size):
    # read frame, resize, convert to gray
    ret, frame_bgr = cap.read()
    if not ret:
        raise RuntimeError
    if im_size is not None:
        frame_bgr = cv2.resize(frame_bgr, im_size)
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return frame_bgr, frame

def get_flow_cpu(previous_frame, current_frame, hsv):

    # calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(
        previous_frame, current_frame, None, 0.5, 5, 15, 3, 5, 1.2, 0,
    )

    # convert from cartesian to polar coordinates to get magnitude and angle
    magnitude, angle = cv2.cartToPolar(
        flow[..., 0], flow[..., 1], angleInDegrees=True,
    )

    # set hue according to the angle of optical flow
    hsv[..., 0] = angle * ((1 / 360.0) * (180 / 255.0))

    # set value according to the normalized magnitude of optical flow
    hsv[..., 2] = cv2.normalize(
        magnitude, None, 0.0, 1.0, cv2.NORM_MINMAX, -1,
    )

    # multiply each pixel value to 255
    hsv_8u = np.uint8(hsv * 255.0)

    # convert hsv to bgr
    bgr_cpu = cv2.cvtColor(hsv_8u, cv2.COLOR_HSV2BGR)

    return bgr_cpu

def get_flow_gpu(gpu_previous, gpu_current, gpu_h, gpu_s, gpu_hsv, gpu_hsv_8u):

    # create optical flow instance
    gpu_flow = cv2.cuda_FarnebackOpticalFlow.create(
        5, 0.5, False, 15, 3, 5, 1.2, 0,
    )
    # calculate optical flow
    gpu_flow = cv2.cuda_FarnebackOpticalFlow.calc(
        gpu_flow, gpu_previous, gpu_current, None,
    )

    gpu_flow_x = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)
    gpu_flow_y = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)
    cv2.cuda.split(gpu_flow, [gpu_flow_x, gpu_flow_y])

    # convert from cartesian to polar coordinates to get magnitude and angle
    gpu_magnitude, gpu_angle = cv2.cuda.cartToPolar(
        gpu_flow_x, gpu_flow_y, angleInDegrees=True,
    )

    # set value to normalized magnitude from 0 to 1
    gpu_v = cv2.cuda.normalize(gpu_magnitude, 0.0, 1.0, cv2.NORM_MINMAX, -1)

    # get angle of optical flow
    angle = gpu_angle.download()
    angle *= (1 / 360.0) * (180 / 255.0)

    # set hue according to the angle of optical flow
    gpu_h.upload(angle)

    # merge h,s,v channels
    cv2.cuda.merge([gpu_h, gpu_s, gpu_v], gpu_hsv)

    # multiply each pixel value to 255
    gpu_hsv.convertTo(cv2.CV_8U, 255.0, gpu_hsv_8u, 0.0)

    # convert hsv to bgr
    gpu_bgr = cv2.cuda.cvtColor(gpu_hsv_8u, cv2.COLOR_HSV2BGR)

    # send result from GPU back to CPU
    bgr_gpu = gpu_bgr.download()

    return bgr_gpu

def main(video):
    # set up window for results
    window_name = "result"
    get_window(window_name)

    # image size for processing
    im_size = None # (960, 540)

    # init video capture with video
    cap = cv2.VideoCapture(video)

    # get default video FPS
    fps = cap.get(cv2.CAP_PROP_FPS)

    # get total number of video frames
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # read the first frame
    cpu_frame_bgr, old_cpu_frame_gray = get_next_frame(cap, im_size)

    # create hsv output for optical flow and set saturation to 1
    cpu_hsv = np.zeros_like(cpu_frame_bgr, np.float32)
    cpu_hsv[..., 1] = 1.0

    if HAS_GPU:

        # upload resized frame to GPU
        gpu_frame_bgr = cv2.cuda_GpuMat()
        gpu_frame_bgr.upload(cpu_frame_bgr)

        # convert to gray
        old_cpu_frame_gray = cv2.cvtColor(cpu_frame_bgr, cv2.COLOR_BGR2GRAY)

        # upload pre-processed frame to GPU
        old_gpu_frame_gray = cv2.cuda_GpuMat()
        old_gpu_frame_gray.upload(old_cpu_frame_gray)

        # create gpu_hsv output for optical flow
        gpu_hsv = cv2.cuda_GpuMat(gpu_frame_bgr.size(), cv2.CV_32FC3)
        gpu_hsv_8u = cv2.cuda_GpuMat(gpu_frame_bgr.size(), cv2.CV_8UC3)

        # separate hsv components (not gpu_v as this gets created on the fly) and set saturation to 1
        gpu_h = cv2.cuda_GpuMat(gpu_frame_bgr.size(), cv2.CV_32FC1)
        gpu_s = cv2.cuda_GpuMat(gpu_frame_bgr.size(), cv2.CV_32FC1)
        gpu_s.upload(np.ones_like(old_cpu_frame_gray, np.float32))


    for _ in range(min(num_frames, 10)):

        # read the next frame
        cpu_frame_bgr, new_cpu_frame_gray = get_next_frame(cap, im_size)

        # get CPU flow
        bgr_cpu = get_flow_cpu(old_cpu_frame_gray, new_cpu_frame_gray, cpu_hsv)

        # update previous_frame value
        old_cpu_frame_gray = new_cpu_frame_gray

        if HAS_GPU:

            # upload frame to GPU
            gpu_frame_bgr.upload(cpu_frame_bgr)

            # convert to gray
            new_gpu_frame_gray = cv2.cuda.cvtColor(gpu_frame_bgr, cv2.COLOR_BGR2GRAY)

            # get GPU flow
            bgr_gpu = get_flow_gpu(old_gpu_frame_gray, new_gpu_frame_gray, gpu_h, gpu_s, gpu_hsv, gpu_hsv_8u)

            # update previous_frame value
            old_gpu_frame_gray = new_gpu_frame_gray

        # visualization
        result = cv2.hconcat([cpu_frame_bgr, bgr_cpu])
        if HAS_GPU:
            result = cv2.hconcat([result, bgr_gpu])
        cv2.imshow(window_name, result)
        k = cv2.waitKey(1)
        if k == 27:
            break

    # release the capture
    cap.release()

    # destroy all windows
    cv2.destroyAllWindows()

    # print results
    print("Number of frames : ", num_frames)


if __name__ == "__main__":

    # init argument parser
    parser = argparse.ArgumentParser(description="OpenCV CPU/GPU Comparison")

    parser.add_argument(
        "--video", help="path to .mp4 video file", required=True, type=str,
    )

    # parsing script arguments
    args = parser.parse_args()
    video = args.video

    # output passed arguments
    print("Configuration")
    print("- video file : ", video)

    # run pipeline
    main(video)