#include <iostream>

#include <opencv2/opencv.hpp>
#include <MNN/Tensor.hpp>

#include "retinaface.hpp"

int main(int, char**) {
	
    std::cout << "Hello, world!\n";

    cv::Mat image = cv::imread("/Users/vincent/Documents/Repo/RetinaFace_MNN/test.jpg");
    RetinaFace detector("/Users/vincent/Documents/Repo/RetinaFace_MNN/retinaface.mnn");

    std::vector<BBox> final_bboxes;

    detector.detect(image, final_bboxes);


}
