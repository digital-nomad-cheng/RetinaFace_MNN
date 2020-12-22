
// Reference: https://github.com/biubug6/Face-Detector-1MB-with-landmark/tree/master/Face_Detector_ncnn
class RetinaFace
{
public:
	RetinaFace(const std::string& model_path, const std::string& model_name);
	~RetinaFace();

	void detect(const cv::Mat& image, std::vector<BBox>& final_bboxes) const;
private:
	
};