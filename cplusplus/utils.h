#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/videoio.hpp>

cv::VideoCapture open_vid(const std::string &video_fname);

void get_next_frame(cv::VideoCapture &cap, cv::Mat &frame, const cv::Size &size);

at::Tensor to_tensor(const cv::Mat &img, const at::Device &device, const bool channel_first=true, const bool unsqueeze=true);

void show_image(const cv::Mat& img, const std::string &title);

cv::Mat to_cvmat(const at::Tensor &tensor, const bool channel_last=true, const bool squeeze=true);

torch::jit::script::Module load_module(const std::string &fname, const at::Device &device, const bool eval=true);

at::Tensor infer(torch::jit::script::Module &module, const at::Tensor &input);
