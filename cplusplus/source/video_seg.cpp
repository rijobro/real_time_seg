#include <iostream>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

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

    // Read the image file
    cv::VideoCapture cap(video_fname.c_str(), cv::CAP_ANY);

    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;

    std::cout << "\nfinished!\n";
    return 0;
}
