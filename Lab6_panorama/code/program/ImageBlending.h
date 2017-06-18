#pragma once

#include <opencv2/core/core.hpp>
#include <vector>
#include <utility>

namespace vfx {

class ImageBlending {
public:
  ImageBlending(std::vector<cv::Mat>&& images,
                std::vector<float>&& focal,
                std::vector<cv::Matx<float,3,3>>&& transforms);
  ImageBlending(ImageBlending&& other);
  void process();
  void getResult(cv::OutputArray output);
  
private:
  std::vector<cv::Mat> m_Images;
  std::vector<float> m_FocalLengths;
  std::vector<cv::Matx<float,3,3>> m_Transforms; // From framebuffer to source image
  
  struct TransformedRect {
    cv::Rect bound;
    cv::Point2f vertex[4];
  };
  
  std::vector<TransformedRect> m_Bounds;
  
  cv::Mat framebuffer;
  cv::Mat stencilbuffer;
  
};

}