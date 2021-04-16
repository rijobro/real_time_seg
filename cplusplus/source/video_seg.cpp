#include <iostream>
#include <torch/torch.h>
// #include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

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

    // Read the image file
    Mat frame;
    VideoCapture cap;
    cap.open(video_fname.c_str(), cv::CAP_ANY);
    // check if we succeeded
    if (!cap.isOpened()) {
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

    cout << "Start grabbing" << endl
        << "Press any key to terminate" << endl;
    for (unsigned i=0;;++i) {
        cout << "\nreading frame " << i << endl;
        // wait for a new frame from camera and store it into 'frame'
        cap.read(frame);
        // check if we succeeded
        if (frame.empty()) {
            cerr << "ERROR! blank frame grabbed\n";
            break;
        }
    }

    torch::Tensor tensor = torch::rand({2, 3});
    cout << tensor << endl;

    cout << "\nfinished!\n";
    return 0;
}
