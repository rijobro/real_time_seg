#include "utils.h"
#include <opencv2/opencv.hpp>

cv::VideoCapture open_vid(const std::string &video_fname)
{
    cv::VideoCapture cap;
    cap.open(video_fname.c_str(), cv::CAP_ANY);
    if (cap.isOpened())
        return cap;
    // failed to open video
    std::cerr << "Error, failed to open video\n";
    exit(1);
}

void get_next_frame(cv::VideoCapture &cap, cv::Mat &frame, const unsigned frame_num)
{
    // get frame
    cap.read(frame);

    // check if we succeeded
    if (frame.empty()) {
        std::cerr << "Error, failed to grab frame " << frame_num << ".\n";
        exit(1);
    }
}

int round_up(const unsigned num, const unsigned multiple)
{
    assert(multiple);
    return ((num + multiple -1) / multiple) * multiple;
}

torch::nn::ZeroPad2d get_padder(const cv::VideoCapture &cap, const unsigned divisible_factor)
{
    const unsigned in_w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    const unsigned in_h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    const unsigned out_w = round_up(in_w, divisible_factor);
    const unsigned out_h = round_up(in_h, divisible_factor);

    const unsigned pad_w = out_w - in_w;
    const unsigned pad_h = out_h - in_h;
    const unsigned pad_t = pad_h / 2;
    const unsigned pad_b = pad_h - pad_t;
    const unsigned pad_l = pad_w / 2;
    const unsigned pad_r = pad_w - pad_l;

    return torch::nn::ZeroPad2d(torch::nn::ZeroPad2dOptions({pad_l, pad_r, pad_t, pad_b}));
}

at::Tensor to_tensor(const cv::Mat &img, const at::Device &device, torch::nn::ZeroPad2d &padder)
{
    at::Tensor tensor_image = torch::from_blob(img.data, { img.rows, img.cols, 3 }, at::kByte);
    tensor_image = tensor_image.to(at::kFloat);

    // to device
    tensor_image = tensor_image.to(device);

    // move channel from end to first
    tensor_image = tensor_image.movedim(-1, 0);

    // unsqueeze
    tensor_image.unsqueeze_(0);

    tensor_image = padder->forward(tensor_image);

    return tensor_image;
}

void show_image(const cv::Mat& img, const std::string &title)
{
    cv::namedWindow(title, cv::WINDOW_NORMAL); // Create a window for display.
    cv::imshow(title, img);
    cv::waitKey(0);
}

cv::Mat to_cvmat(const at::Tensor &tensor)
{
    at::Tensor tensor_mod = tensor;

    // squeeze first channel
    tensor_mod = tensor_mod.squeeze(0);

    // move channel to end
    tensor_mod = tensor_mod.movedim(0, -1);

    const int width = tensor_mod.sizes()[0];
    const int height = tensor_mod.sizes()[1];
    try
    {
        cv::Mat output_mat(cv::Size{ height, width }, CV_8UC3, tensor_mod.to(at::kByte).data_ptr<uchar>());
        return output_mat;
    }
    catch (const c10::Error& e)
    {
        std::cout << "an error has occured : " << e.msg() << std::endl;
    }
    return cv::Mat(height, width, CV_8UC3);
}

torch::jit::script::Module load_module(const std::string &fname, const at::Device &device)
{
    torch::jit::script::Module module;
    module = torch::jit::load(fname);
    module.eval();
    torch::NoGradGuard no_grad;
    module.to(device);
    return module;
}

at::Tensor infer(torch::jit::script::Module &module, const at::Tensor &input)
{
    return module.forward(std::vector<torch::jit::IValue>({input})).toTensor();
}
