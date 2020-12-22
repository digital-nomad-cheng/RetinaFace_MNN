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
    preproc_config.filterType = MNN::CV::NEAREST;
    ::memcpy(preproc_config.mean, this->_mean_vals, sizeof(this->_mean_vals));
    // no norm in this model
    // ::memcpy(preproc_config.normal, norm_vals, sizeof(norm_vals));
    preproc_config.sourceFormat = MNN::CV::BGR;
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

    float *scores = this->_output_cls_tensor->host<float>();
    float *offsets = this->_output_bbox_tensor->host<float>();
    float *ldmks = this->_output_ldmk_tensor->host<float>();

    for (const Box& anchor : anchors) {
        BBox bbox;
        Box refined_box;
        
        if (scores[1] > this->_score_threshold) {
            // score
            bbox.score = scores[1];

            // bbox
            refined_box.cx = anchor.cx + offsets[0] * 0.1 * anchor.sx;
            refined_box.cy = anchor.cy + offsets[1] * 0.1 * anchor.sy;
            refined_box.sx = anchor.sx * exp(offsets[2] * 0.2);
            refined_box.sy = anchor.sy * exp(offsets[3] * 0.2);

            bbox.x1 = (refined_box.cx - refined_box.sx/2) * image.cols;
            bbox.y1 = (refined_box.cy - refined_box.sy/2) * image.rows;
            bbox.x2 = (refined_box.cx + refined_box.sx/2) * image.cols;
            bbox.y2 = (refined_box.cy + refined_box.sy/2) * image.rows;

            clip_bboxes(bbox, image.cols, image.rows);

            // landmarks
            for (int i = 0; i < 5; i++) {
                bbox.landmarks[i].x = (anchor.cx + ldmks[2*i] * 0.1 * anchor.sx) * image.cols;
                bbox.landmarks[i].y = (anchor.cy + ldmks[2*i+1] * 0.1 * anchor.sy) * image.rows;
            }

            final_bboxes.push_back(bbox);
        }

        scores += 2;
        offsets += 4;
        ldmks += 10;
    }

    std::sort(final_bboxes.begin(), final_bboxes.end(), [](BBox &lsh, BBox &rsh) {
        return lsh.score > rsh.score;
    });
    nms(final_bboxes, this->_nms_threshold);
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

void RetinaFace::nms(std::vector<BBox>& bboxes, float nms_threshold) const
{
    std::vector<float> bbox_areas(bboxes.size());
    for (int i = 0; i < bboxes.size(); i++) {
        bbox_areas[i] = (bboxes.at(i).x2 - bboxes.at(i).x1 + 1) * (bboxes.at(i).y2 - bboxes.at(i).y1 + 1);
    }

    for (int i = 0; i < bboxes.size(); i++) {
        for (int j = i + 1; j < bboxes.size(); ) {
            float xx1 = std::max(bboxes[i].x1, bboxes[j].x1);
            float yy1 = std::max(bboxes[i].y1, bboxes[j].y1);
            float xx2 = std::min(bboxes[i].x2, bboxes[j].x2);
            float yy2 = std::min(bboxes[i].y2, bboxes[j].y2);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float h = std::max(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float IoU = inter / (bbox_areas[i] + bbox_areas[j] - inter);
            if (IoU >= nms_threshold) {
                bboxes.erase(bboxes.begin() + j);
                bbox_areas.erase(bbox_areas.begin() + j);
            }
            else {
                j++;
            }
        }
    }
}

inline void RetinaFace::clip_bboxes(BBox& bbox, int w, int h) const
{
    if(bbox.x1 < 0) bbox.x1 = 0;
    if(bbox.y1 < 0) bbox.y1 = 0;
    if(bbox.x2 > w) bbox.x2 = w;
    if(bbox.y2 > h) bbox.y2 = h;
}
