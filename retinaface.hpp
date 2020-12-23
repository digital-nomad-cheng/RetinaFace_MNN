
// Reference: https://github.com/biubug6/Face-Detector-1MB-with-landmark/tree/master/Face_Detector_ncnn

#include <string>

#include <opencv2/opencv.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/ImageProcess.hpp>

struct Point {
    float x;
    float y;
};

struct Box
{
    /* 
     * cx: x of box center
     * cy: y of box center
     * sx: width of box
     * sy: height of box
     */

    float cx;
    float cy;
    float sx;
    float sy;
};

struct BBox
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    Point landmarks[5];
};

class RetinaFace
{
public:
    RetinaFace(const std::string& model_file);
    ~RetinaFace();

    void detect(const cv::Mat& image, std::vector<BBox>& final_bboxes) const;

private:
    void create_anchors(std::vector<Box>& anchors, int w, int h) const;
    void nms(std::vector<BBox>& input_bboxes, float nms_threshold=0.5) const;
    void clip_bboxes(BBox& bbox, int w, int h) const;
	
    float _nms_threshold = 0.4;
    float _score_threshold = 0.6;
    const float _mean_vals[3] = {104.f, 117.f, 123.f};

    const int _in_w = 320;
    const int _in_h = 480;

    std::shared_ptr<MNN::Interpreter> _net;
    MNN::Session *_net_sess;
    MNN::Tensor *_input_tensor = nullptr;
    MNN::Tensor *_output_cls_tensor = nullptr;
    MNN::Tensor *_output_bbox_tensor = nullptr;
    MNN::Tensor *_output_ldmk_tensor = nullptr;

    std::shared_ptr<MNN::CV::ImageProcess> pretreat_data;

    const int num_threads = 4;

    std::vector<Box> anchors;

};