#pragma once
#include <opencv2/core/core.hpp>
#include <vector>
#include <utility>
#include <string>
#include "Graph.h"
#include "ImageBlending.h"
#include "FeatureMatching.h"

namespace vfx {

class CylindricalPanorama {
public:
  CylindricalPanorama();
  ~CylindricalPanorama();
  
  void addImage(cv::Mat image, float focal);
  void process();
  void getResult(std::vector<cv::Mat>& output);
  
private:
  void processFeatureMatching();
  void buildImageGraph();
  
private:
  std::vector<float> m_FocalLengths;
  std::vector<cv::Mat> m_InputImages;
  std::vector<std::pair<int,int>> m_ImagePairs;
  std::vector<cv::Matx<float,3,3>> m_Homographies;
  
  //std::vector<std::vector<int>> m_ImageSets;
  Graph<cv::Matx<float,3,3>> m_ImageGraph;
  std::vector<ImageBlending> m_Panoramas;
  
};

extern TransformationType g_TransformationType;
extern bool g_EnableWarping;
extern bool g_DrawBoundingRects;
extern bool g_ShowRansacResult;
extern std::string g_OutputDirectory;
extern std::string g_RansacOutputDirectory;

}
