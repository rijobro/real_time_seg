#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/videoio.hpp>

cv::VideoCapture open_vid(const std::string &video_fname);

void get_next_frame(cv::VideoCapture &cap, cv::Mat &frame, const unsigned frame_num);

torch::nn::ZeroPad2d get_padder(const cv::VideoCapture &cap, const unsigned divisible_factor);

at::Tensor to_tensor(const cv::Mat &img, const at::Device &device, torch::nn::ZeroPad2d &padder);

void show_image(const cv::Mat& img, const std::string &title);

cv::Mat to_cvmat(const at::Tensor &tensor);

torch::jit::script::Module load_module(const std::string &fname, const at::Device &device);

at::Tensor infer(torch::jit::script::Module &module, const at::Tensor &input);
