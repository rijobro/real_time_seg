#include "utils.h"
#include <torch/script.h>

void parse_input(const int argc, char** argv, std::string &video_fname, std::string &model_path)
{
    const int num_arguments = argc-1;
    if (num_arguments != 2) {
        std::cout << "\nusage: " << argv[0] << " video_fname model_path\n";
        exit(1);
    }

    video_fname = argv[1];
    model_path = argv[2];

    std::cout << "\nvideo_fname: " << video_fname << "\n";
    std::cout << "model_path: " << model_path << "\n";
}

int main(int argc, char** argv)
{
    // get input params
    std::string video_fname, model_path;
    parse_input(argc, argv, video_fname, model_path);

    // determine device
    const at::Device device(torch::cuda::is_available() ? "cuda:0" : "cpu");
    std::cout << "\ndevice: " << device << "\n";

    // load torch model
    torch::jit::script::Module module = load_module(model_path, device);

    // read the video file
    cv::Mat frame;
    cv::VideoCapture cap = open_vid(video_fname);

    // loop over frames
    const int num_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    for (int i=0; i<num_frames; ++i) {
        std::cout << "\nprocessing frame " << i << " of " << num_frames << "\n";

        // get next frame and convert to torch tensor
        get_next_frame(cap, frame, cv::Size(1264, 1024));
        const at::Tensor tensor = to_tensor(frame, device);
        std::cout << "\ntensor dtype: " << tensor.dtype() << "\n";

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

    std::cout << "\nfinished processing " << num_frames << " frames!\n";
    return 0;
}
