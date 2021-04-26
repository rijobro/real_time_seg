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

    return bgr_cpu, flow

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

def warp_flow_cpu(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

def add_text(image, texts):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = font_thickness = 3
    font_color = (0, 255, 26)
    for row in range(len(texts)):
        for col in range(len(texts[0])):
            text = texts[row][col]
            x = int(image.shape[1] / len(texts[0]) * col)
            y = int(image.shape[0] / len(texts) * row) + 100
            image = cv2.putText(
                image, text, (x,y), font, font_size, font_color, font_thickness, cv2.LINE_AA)
    return image



def main(video, seg):
    # set up window for results
    window_name = "result"
    get_window(window_name)

    # image size for processing
    im_size = None # (960, 540)

    # init video capture with video
    cap = cv2.VideoCapture(video)
    cap_seg = cv2.VideoCapture(seg)

    # get default video FPS
    fps = cap.get(cv2.CAP_PROP_FPS)

    # get total number of video frames
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frames_seg = int(cap_seg.get(cv2.CAP_PROP_FRAME_COUNT))
    if num_frames != num_frames_seg:
        raise RuntimeError

    # read the first frame
    old_cpu_frame_bgr, old_cpu_frame_gray = get_next_frame(cap, im_size)
    old_cpu_frame_seg_bgr, _ = get_next_frame(cap_seg, im_size)

    # create hsv output for optical flow and set saturation to 1
    cpu_hsv = np.zeros_like(old_cpu_frame_bgr, np.float32)
    cpu_hsv[..., 1] = 1.0

    if HAS_GPU:

        # upload resized frame to GPU
        gpu_frame_bgr = cv2.cuda_GpuMat()
        gpu_frame_bgr.upload(old_cpu_frame_bgr)

        # convert to gray
        old_cpu_frame_gray = cv2.cvtColor(old_cpu_frame_bgr, cv2.COLOR_BGR2GRAY)

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


    for f in range(1, num_frames):

        # read the next frame
        cpu_frame_bgr, new_cpu_frame_gray = get_next_frame(cap, im_size)
        new_cpu_frame_seg_bgr, _ = get_next_frame(cap_seg, im_size)

        # get CPU flow
        bgr_cpu, flow_cpu = get_flow_cpu(old_cpu_frame_gray, new_cpu_frame_gray, cpu_hsv)

        pred_cpu_frame_seg_bgr = warp_flow_cpu(old_cpu_frame_seg_bgr, flow_cpu)

        if HAS_GPU:

            # upload frame to GPU
            gpu_frame_bgr.upload(cpu_frame_bgr)

            # convert to gray
            new_gpu_frame_gray = cv2.cuda.cvtColor(gpu_frame_bgr, cv2.COLOR_BGR2GRAY)

            # get GPU flow
            bgr_gpu = get_flow_gpu(old_gpu_frame_gray, new_gpu_frame_gray, gpu_h, gpu_s, gpu_hsv, gpu_hsv_8u)

        # visualization
        to_imshow = [[old_cpu_frame_bgr, cpu_frame_bgr, bgr_cpu],
                     [old_cpu_frame_seg_bgr, new_cpu_frame_seg_bgr, pred_cpu_frame_seg_bgr]]
        names = [[f"prev. frame ({f-1})", f"current frame ({f})", "opt. flow"],
                 [f"prev. label ({f-1})", f"current label ({f})", "prev. warped w/ opt. flow"]]
        result = cv2.vconcat([cv2.hconcat(list_h) for list_h in to_imshow])
        if HAS_GPU:
            result = cv2.hconcat([result, bgr_gpu])
        result = add_text(result, names)
        cv2.imshow(window_name, result)
        # if f > 28:
            # input("Press any key to advance to next image.")
        k = cv2.waitKey(1)
        if k == 27:
            break

        # update previous_frame value
        old_cpu_frame_gray = new_cpu_frame_gray
        old_cpu_frame_bgr = cpu_frame_bgr
        old_cpu_frame_seg_bgr = new_cpu_frame_seg_bgr
        if HAS_GPU:
            old_gpu_frame_gray = new_gpu_frame_gray

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
        "--video", help="path to .mp4 video file", type=str,
        default="./data/d1_im.mp4",
    )

    parser.add_argument(
        "--seg", help="path to segmented .mp4 video file", type=str,
        default="./data/d1_seg.mp4",
    )

    # parsing script arguments
    args = parser.parse_args()
    video = args.video
    seg = args.seg

    # output passed arguments
    print("Configuration")
    print("- video file : ", video)
    print("- seg file : ", seg)

    # run pipeline
    main(video, seg)