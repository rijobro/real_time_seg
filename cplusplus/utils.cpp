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

void get_next_frame(cv::VideoCapture &cap, cv::Mat &frame, const cv::Size &size)
{
    // get frame
    cap.read(frame);

    // check if we succeeded
    if (frame.empty()) {
        std::cerr << "Error, failed to grab frame.\n";
        exit(1);
    }

    // resize
    cv::resize(frame, frame, size);
}

at::Tensor to_tensor(const cv::Mat &img, const at::Device &device, const bool channel_first, const bool unsqueeze)
{
    at::Tensor tensor_image = torch::from_blob(img.data, { img.rows, img.cols, 3 }, at::kByte);
    tensor_image = tensor_image.to(at::kFloat);

    // to device
    tensor_image = tensor_image.to(device);

    // move channel from end to first
    if (channel_first)
        tensor_image = tensor_image.movedim(-1, 0);

    // if desired, unsqueeze
    if (unsqueeze)
        tensor_image.unsqueeze_(0);

    return tensor_image;
}

void show_image(const cv::Mat& img, const std::string &title)
{
    cv::namedWindow(title, cv::WINDOW_NORMAL); // Create a window for display.
    cv::imshow(title, img);
    cv::waitKey(0);
}

cv::Mat to_cvmat(const at::Tensor &tensor, const bool channel_last, const bool squeeze)
{
    at::Tensor tensor_mod = tensor;

    // if desired, squeeze first channel
    if (squeeze)
        tensor_mod = tensor_mod.squeeze(0);

    // if desired, move channel to end
    if (channel_last)
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

torch::jit::script::Module load_module(const std::string &fname, const at::Device &device, const bool eval)
{
    torch::jit::script::Module module;
    module = torch::jit::load(fname);
    if (eval)
        module.eval();
    module.to(device);
    return module;
}

at::Tensor infer(torch::jit::script::Module &module, const at::Tensor &input)
{
    return module.forward(std::vector<torch::jit::IValue>({input})).toTensor();
}
