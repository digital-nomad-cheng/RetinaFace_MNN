#include "retinaface.hpp"

RetinaFace::RetinaFace(const std::string& model_file)
{
    this->_net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_file.data()));
    if (_net == nullptr) {
        std::cout << "Failed to load model from path: " <<  model_file << std::endl;
    } else {
        std::cout << "Successfully to load model from path: " << model_file << std::endl;
    }

    // sess config
    MNN::ScheduleConfig sess_config;
    sess_config.type = MNN_FORWARD_CPU;
    sess_config.numThread = this->num_threads;

    MNN::BackendConfig backend_config;
    backend_config.precision = backend_config.Precision_High;
    backend_config.power = backend_config.Power_High;
    sess_config.backendConfig = &backend_config;

    // create session
    _net_sess = _net->createSession(sess_config);

    // config input and output tensors
    _input_tensor = _net->getSessionInput(_net_sess, "input");
    _output_cls_tensor = _net->getSessionOutput(_net_sess, "cls");
    _output_bbox_tensor = _net->getSessionOutput(_net_sess, "bbox");
    _output_ldmk_tensor = _net->getSessionOutput(_net_sess, "ldmk");

    // create image preprocessing pipeline
    MNN::CV::ImageProcess::Config preproc_config;
    preproc_config.filterType = MNN::CV::BILINEAR;
    ::memcpy(preproc_config.mean, this->_mean_vals, sizeof(this->_mean_vals));
    // no norm in this model
    // ::memcpy(preproc_config.normal, norm_vals, sizeof(norm_vals));
    preproc_config.sourceFormat = MNN::CV::RGB;
    preproc_config.destFormat = MNN::CV::RGB;
    this->pretreat_data = std::shared_ptr<MNN::CV::ImageProcess>(MNN::CV::ImageProcess::create(preproc_config));
}

RetinaFace::~RetinaFace()
{
    this->_net->releaseModel();
}

void RetinaFace::detect(const cv::Mat& image, std::vector<BBox>& final_bboxes) const
{
    cv::imshow("image", image);
    cv::waitKey(0);

    // forward inference
    // resize session according input shape
    std::vector<int> input_dims = {1, 3, image.rows, image.cols};
    _net->resizeTensor(this->_input_tensor, input_dims);
    _net->resizeSession(_net_sess);

    // preprocess image
    MNN::CV::Matrix trans;
    trans.postScale(1.0f/image.cols, 1.0f/image.rows);
    trans.postScale(image.cols, image.rows);
    pretreat_data->setMatrix(trans);
    pretreat_data->convert((uint8_t*)image.data, image.cols, image.rows, 0, this->_input_tensor);
    
    _net->runSession(_net_sess);

    auto output_shape = this->_output_cls_tensor->shape();
    std::cout << "output_shape[0]: " << output_shape[0] <<
        " output_shape[1]: " << output_shape[1] << 
        " output_shape[2]: " << output_shape[2] << std::endl;

    std::vector<Box> anchors;
    this->create_anchors(anchors, image.cols, image.rows);
    std::cout << "anchors size: " << anchors.size() << std::endl;

    for (const Box& anchor : anchors) {
        BBox bbox;

    }


}   

void RetinaFace::create_anchors(std::vector<Box>& anchors, int w, int h) const
{
    // create predefined anchors
    anchors.clear();
    std::vector<std::vector<int> > feature_map(3), min_sizes(3);
    float steps[] = {8, 16, 32};
    for (int i = 0; i < feature_map.size(); ++i) {
        feature_map[i].push_back(ceil(h/steps[i]));
        feature_map[i].push_back(ceil(w/steps[i]));
    }
    std::vector<int> minsize1 = {10, 20};
    min_sizes[0] = minsize1;
    std::vector<int> minsize2 = {32, 64};
    min_sizes[1] = minsize2;
    std::vector<int> minsize3 = {128, 256};
    min_sizes[2] = minsize3;
    
    for (int k = 0; k < feature_map.size(); ++k) {
        std::vector<int> min_size = min_sizes[k];
        for (int i = 0; i < feature_map[k][0]; ++i) {
            for (int j = 0; j < feature_map[k][1]; ++j) {
                for (int l = 0; l < min_size.size(); ++l) {
                    float s_kx = min_size[l]*1.0/w;
                    float s_ky = min_size[l]*1.0/h;
                    float cx = (j + 0.5) * steps[k]/w;
                    float cy = (i + 0.5) * steps[k]/h;
                    Box axil = {cx, cy, s_kx, s_ky};
                    anchors.push_back(axil);
                }
            }
        }

    }

}

void RetinaFace::nms(std::vector<BBox> &input_boxes, float nms_threshold) const
{
    std::vector<float>vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        for (int j = i + 1; j < int(input_boxes.size());) {
            float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float h = std::max(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= nms_threshold) {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else {
                j++;
            }
        }
    }
}