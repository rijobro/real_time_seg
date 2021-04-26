#include <torch/torch.h>
#include <torch/script.h>

#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

cv::VideoCapture open_vid(const std::string &video_fname)
{
    cv::VideoCapture cap;
    cap.open(video_fname.c_str(), cv::CAP_ANY);
    if (cap.isOpened())
        return cap;
    // failed to open video
    std::cerr << "ERROR! Unable to open camera\n";
    exit(1);
}

void get_next_frame(cv::VideoCapture &cap, cv::Mat &frame, const cv::Size &size)
{
    // get frame
    cap.read(frame);

    // check if we succeeded
    if (frame.empty()) {
        std::cerr << "ERROR! blank frame grabbed\n";
        exit(1);
    }

    // resize
    cv::resize(frame, frame, size);
}

const at::Tensor cv2mat_to_tensor(const cv::Mat &img, const at::Device &device, const bool channel_first=true, const bool unsqueeze=true)
{
    // opencv -> torch
    at::Tensor tensor_image = torch::from_blob(img.data, { img.rows, img.cols, 3 }, at::kFloat);

    // to device
    tensor_image.to(device);

    // move channel from end to first
    if (channel_first)
        tensor_image = tensor_image.movedim(-1, 0);

    // if desired, unsqueeze
    if (unsqueeze)
        tensor_image.unsqueeze_(0);

    return tensor_image;
}

int main(int argc, char** argv)
{
    const int num_arguments = argc-1;
    if (num_arguments != 2) {
        std::cout << "\nusage: " << argv[0] << " video_fname model_path\n";
        exit(1);
    }
    const std::string video_fname = argv[1];
    const std::string model_path = argv[2];
    std::cout << "\nvideo_fname: " << video_fname << "\n";
    std::cout << "model_path: " << model_path << "\n";

    // determine device
    const at::Device device(torch::cuda::is_available() ? "cuda:0" : "cpu");
//    const at::Device device("cpu");
    std::cout << "\ndevice: " << device << "\n";

    // load torch model
    torch::jit::script::Module module;
    module = torch::jit::load(model_path);
    module.to(at::Device("cpu"));

    // Read the video file
    cv::Mat frame;
    cv::VideoCapture cap = open_vid(video_fname);

    // loop over frames
    const int num_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    for (int i=0; i<num_frames; ++i) {

        // get next frame and convert to torch tensor
        get_next_frame(cap, frame, cv::Size(1264, 1024));
        const at::Tensor tensor = cv2mat_to_tensor(frame, device);

        // first time, print info
        if (i==0) {
            std::cout << "image shape: " << frame.size() << std::endl;
            std::cout << "tensor shape: " << tensor.sizes() << std::endl;
        }

        // infer
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(tensor);

        at::Tensor output = module.forward(inputs).toTensor();
        if (i==0)
            std::cout << "output shape: " << output.sizes() << std::endl;
        exit(0);
    }

    std::cout << "\nfinished processing " << num_frames << " frames!\n";
    return 0;
}
