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

    std::cout << "total faces:" << final_bboxes.size() << std::endl;
    for (BBox& bbox : final_bboxes) {
    	cv::rectangle(image, cv::Point(bbox.x1, bbox.y1), cv::Point(bbox.x2, bbox.y2), cv::Scalar(255, 0, 0), 2);
    }
    
    cv::imshow("image", image);
    cv::waitKey(0);

}
