#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

namespace vfx {

const int FEATURE_DESC_WIDTH = 8;
const int FEATURE_DESC_SIZE = 64;
const int FEATURE_WINDOW_WIDTH = 40;

struct FeatureInfo {
  cv::Point2f position;
  float descriptor[FEATURE_DESC_SIZE];
};

typedef std::vector<FeatureInfo> FeatureInfoList;

class FeatureDetection {
public:
  FeatureDetection();
  ~FeatureDetection();
  
  void init(cv::Mat image);
  void process();
  void getFeatures(FeatureInfoList& output);
  void showResult(const char* title = nullptr);
  
private:
  struct FeatureData {
    cv::Point2f position;
    int level;
    float response;
    bool valid;
    
    float orientation;
    float descriptor[FEATURE_DESC_SIZE];
  };
  
  typedef std::vector<FeatureData> FeatureList;
  
private:
  void constructPyramid(cv::Mat greyimage);
  void showPyramid();
  
  void computeHarrisResponse(int level, cv::Mat& result);
  void markFeatures(cv::Mat response, cv::Mat& result, float threshold);
  void extractFeatures(int level, cv::Mat featureMask,cv::Mat response, FeatureList& features);
  void refineFeatures(int level, cv::Mat response, FeatureList& features);
  void removeNonMaximal(FeatureList& features);
  
  void computeOrientation(FeatureList& features);
  void computeDescriptor(FeatureList& features);
  
  cv::Matx<float,2,3> getTransformation(const FeatureData& feature);
  void draw(cv::Mat canvas, const FeatureData& feature);
  
private:
  cv::Mat m_SourceImage;
  std::vector<cv::Mat> m_Pyramid;
  std::vector<FeatureData> m_FeatureList;
};

}
