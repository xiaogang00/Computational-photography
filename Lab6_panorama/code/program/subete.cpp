#include "CylindricalPanorama.h"
#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include <opencv2/highgui/highgui.hpp>

static std::vector<std::string> s_InputPaths;
static std::vector<float> s_FocalLengths;

static void load_images(const char* listfile);

int main(int argc, char* argv[])
{
  if (argc < 2) {
    std::exit(1);
  }
  load_images(argv[1]);
  
  for (int i = 2; i < argc; ++i) {
    if (strcmp(argv[i], "--no-warping") == 0) {
      vfx::g_EnableWarping = false;
      printf("disable warping...\n");
    } else if (strcmp(argv[i], "--show-bounding-rects") == 0) {
      vfx::g_DrawBoundingRects = true;
    }
  }
  
  vfx::CylindricalPanorama pano;
  
  for (auto it = s_InputPaths.begin(); it != s_InputPaths.end(); ++it) {
    int index = it - s_InputPaths.begin();
    float focal = s_FocalLengths[index];
    
    std::cout << "INPUT[" << index << "]: " << *it << std::endl;
    std::cout << "  focal length = " << focal << std::endl;
    
    // Load Image
    cv::Mat image = cv::imread(it->c_str());
    
    pano.addImage(image, focal);
  }
  
  pano.process();
  
  std::vector<cv::Mat> outImages;
  pano.getResult(outImages);
  
  for (auto outImg = outImages.begin(); outImg != outImages.end(); ++outImg) {
    char title[128];
    sprintf(title, "panorama-%d.jpg", (int)(outImg-outImages.begin()));
    cv::imwrite(vfx::g_OutputDirectory + title, *outImg);
    cv::imshow(title, *outImg);
  }
  
  while (cv::waitKey(0) != 27) {
  }
  
  return 0;
}

static void load_images(const char* listfile)
{
  std::ifstream stream;
  stream.open(listfile);
  
  std::string filename;
  float focal_len;
  int num;
  
  stream >> num;
  
  std::string xfmtype;
  stream >> xfmtype;
  if (xfmtype == "HOMO" || xfmtype == "HOMOGRAPHY") {
    vfx::g_TransformationType = vfx::TransformationType::HOMOGRAPHY;
  } else if (xfmtype == "AFFINE") {
    vfx::g_TransformationType = vfx::TransformationType::AFFINE;
  } else if (xfmtype == "TRANS") {
    vfx::g_TransformationType = vfx::TransformationType::TRANSLATION;
  }
  printf("Transformation Type : %s\n", xfmtype.c_str());
  
  for (; num > 0; --num) {
    stream >> filename >> focal_len;
    s_InputPaths.push_back(filename);
    s_FocalLengths.push_back(focal_len);
  }
  
  stream >> vfx::g_RansacOutputDirectory;
  stream >> vfx::g_OutputDirectory;
  
  stream.close();
}
