#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
namespace vfx {

class ImageWarping {
public:
  ImageWarping(float focal_px, float width, float height);
  ImageWarping(float focal_px, cv::Size size);
  
  // From Plane to Warpped Space
  cv::Point2f operator() (cv::Point2f pos) const;
  cv::Mat operator() (cv::Mat image) const;
  
  // From Warpped Space to Plane
  cv::Point2f project(cv::Point2f pos) const;
  
private:
  float m_FocalLength; // In Pixel
  float m_Width;
  float m_Height;
  
};

}
