#include "utils.h"
#include <chrono>
#include <torch/script.h>

void parse_input(const int argc, char** argv, std::string &video_fname, std::string &model_path, at::Device &device)
{
    const int num_arguments = argc-1;
    if (num_arguments < 2) {
        std::cout << "\nusage: " << argv[0] << " video_fname model_path [device]\n";
        exit(1);
    }

    video_fname = argv[1];
    model_path = argv[2];

    if (num_arguments >= 3)
        device = at::Device(argv[3]);
    else
        device = at::Device(torch::cuda::is_available() ? "cuda:0" : "cpu");

    std::cout << "\nvideo_fname: " << video_fname << "\n";
    std::cout << "model_path: " << model_path << "\n";
    std::cout << "\ndevice: " << device << "\n\n";
}

int main(int argc, char** argv)
{
    // get input params
    std::string video_fname, model_path;
    at::Device device("cpu");
    parse_input(argc, argv, video_fname, model_path, device);

    // load torch model
    torch::jit::script::Module module = load_module(model_path, device);

    // read the video file
    cv::Mat frame;
    cv::VideoCapture cap = open_vid(video_fname);
    const int num_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);

    // start clock
    auto t_start = std::chrono::steady_clock::now();

    // loop over frames
    for (int i=0; i<num_frames; ++i) {

        // get next frame and convert to torch tensor
        get_next_frame(cap, frame, cv::Size(1264, 1024));
        const at::Tensor tensor = to_tensor(frame, device);

        // first time, print info
        if (i==0) {
            std::cout << "image shape: " << frame.size() << std::endl;
            std::cout << "tensor shape: " << tensor.sizes() << std::endl;
        }

        // infer
        at::Tensor output = infer(module, tensor);
        if (i==0)
            std::cout << "output shape: " << output.sizes() << std::endl;
    }

    // clock
    auto t_end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    auto fps = float(num_frames) / float(elapsed) * 1000.f;
    std::cout << "processed " << num_frames << " frames at an average of " << fps << " FPS.\n";

    return 0;
}
