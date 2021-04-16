#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

auto ToTensor(cv::Mat img, bool unsqueeze=false, int unsqueeze_dim = 0)
{
    std::cout << "image shape: " << img.size() << std::endl;
    at::Tensor tensor_image = torch::from_blob(img.data, { img.rows, img.cols, 3 }, at::kByte);

    if (unsqueeze)
    {
        tensor_image.unsqueeze_(unsqueeze_dim);
        std::cout << "tensors new shape: " << tensor_image.sizes() << std::endl;
    }
    std::cout << "tensor shape: " << tensor_image.sizes() << std::endl;
    return tensor_image;
}

int main(int argc, char** argv)
{
    const int num_arguments = argc-1;
    if (num_arguments != 2) {
        cout << "\nusage: " << argv[0] << " video_fname model_path\n";
        exit(1);
    }
    const string video_fname = argv[1];
    const string model_path = argv[2];
    cout << "\nvideo_fname: " << video_fname << "\n";
    cout << "model_path: " << model_path << "\n";

    // load torch model
    torch::jit::script::Module module;
    module = torch::jit::load(model_path);
    // try {
        
    // }
    // catch (const c10::Error& e) {
    //     std::cerr << "error loading the model\n";
    //     return -1;
    // }

    // Read the image file
    Mat frame;
    VideoCapture cap;
    cap.open(video_fname.c_str(), cv::CAP_ANY);
    // check if we succeeded
    if (!cap.isOpened()) {
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

    const int num_frames = cap.get(CAP_PROP_FRAME_COUNT);
    for (unsigned i=0; i<num_frames; ++i) {
        cout << "\nreading frame " << i << endl;
        // wait for a new frame from camera and store it into 'frame'
        cap.read(frame);
        // check if we succeeded
        if (frame.empty()) {
            cerr << "ERROR! blank frame grabbed\n";
            return -1;
        }
        auto tensor = ToTensor(frame);
    }

    torch::Tensor tensor = torch::rand({2, 3});
    cout << tensor << endl;

    cout << "\nfinished!\n";
    return 0;
}
