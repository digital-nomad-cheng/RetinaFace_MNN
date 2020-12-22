#include <iostream>

#include <opencv2/opencv.hpp>
#include <MNN/Tensor.hpp>

#include "retinaface.hpp"

int main(int, char**) {
	
    std::cout << "Hello, world!\n";

    RetinaFace detector("/Users/vincent/Documents/Repo/RetinaFace_MNN/retinaface.mnn");
}
