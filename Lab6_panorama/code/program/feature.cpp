#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <opencv2/highgui/highgui.hpp>
#include "FeatureDetection.h"

int main(int argc, char* argv[])
{
  char title[128];
  
  int numImages = argc - 1;
  
  for (int i = 0; i < numImages; ++i) {
    cv::Mat image = cv::imread(argv[i+1]);
    
    vfx::FeatureDetection detector;
    detector.init(image);
    detector.process();
    
    vfx::FeatureInfoList features;
    detector.getFeatures(features);
    
    std::sprintf(title, "features-%d", i);
    detector.showResult(title);
  }
  
  return 0;
}
