#include "ImageWarping.h"
#include <cmath>

namespace vfx {

ImageWarping::ImageWarping(float focal_px, float width, float height)
: m_FocalLength(focal_px), m_Width(width), m_Height(height)
{
}

ImageWarping::ImageWarping(float focal_px, cv::Size size)
#if defined(_MSC_VER) && (_MSC_VER >= 1800)
: m_FocalLength(focal_px), m_Width(size.width), m_Height(size.height)
#else
: ImageWarping(focal_px, size.width, size.height)
#endif
{
}

cv::Point2f ImageWarping::operator() (cv::Point2f pos) const
{
  float f = m_FocalLength;
  float xo = m_Width * 0.5f, yo = m_Height * 0.5f;
  
  float x = pos.x - xo, y = pos.y - yo, s = f;
  
  float xb = s * std::atan2(x, f);
  float yb = s * y / std::sqrt(x*x + f*f);
  
  return cv::Point2f(xb + xo, yb + yo);
}

cv::Mat ImageWarping::operator() (cv::Mat image) const
{
  CV_Assert(image.type() == CV_8UC3 || image.type() == CV_8UC1);
  cv::Mat output(image.size(), image.type(), cv::Scalar(0,0,0));
  
  for (int yw = 0; yw < image.rows; yw++) {
    for (int xw = 0; xw < image.cols; xw++) {
      cv::Point2f texcoord = project(cv::Point2f(xw, yw));
      
      int x0, y0, x1, y1;
      x0 = std::floor(texcoord.x);
      y0 = std::floor(texcoord.y);
      x1 = std::ceil(texcoord.x);
      y1 = std::ceil(texcoord.y);
      
      if ((x0 >= 0 && x1 < image.cols) && (y0 >= 0 && y1 < image.rows)) {
        float tx = texcoord.x - (float)x0;
        float ty = texcoord.y - (float)y0;
        if (image.type() == CV_8UC3) {
          cv::Vec3f c0 = image.at<cv::Vec3b>(y0,x0);
          cv::Vec3f c1 = image.at<cv::Vec3b>(y0,x1);
          cv::Vec3f c2 = image.at<cv::Vec3b>(y1,x0);
          cv::Vec3f c3 = image.at<cv::Vec3b>(y1,x1);
          cv::Vec3f c_1 = c0 + (c1-c0) * tx;
          cv::Vec3f c_2 = c2 + (c3-c2) * tx;
          cv::Vec3f c_3 = c_1 + (c_2-c_1) * ty;
          output.at<cv::Vec3b>(yw,xw) = c_3;
        } else if (image.type() == CV_8UC1) {
          float c0 = image.at<uchar>(y0,x0);
          float c1 = image.at<uchar>(y0,x1);
          float c2 = image.at<uchar>(y1,x0);
          float c3 = image.at<uchar>(y1,x1);
          float c_1 = c0 + (c1-c0) * tx;
          float c_2 = c2 + (c3-c2) * tx;
          float c_3 = c_1 + (c_2-c_1) * ty;
          output.at<uchar>(yw,xw) = c_3;
        }
      }
    }
  }
  
  return output;
}

cv::Point2f ImageWarping::project(cv::Point2f pos) const
{
  float f = m_FocalLength;
  float xo = m_Width * 0.5f, yo = m_Height * 0.5f;
  
  float x = pos.x - xo, y = pos.y - yo, s = f;
  
  float xb = f * std::tan(x / s);
  float yb = y / s * std::sqrt(xb*xb + f*f);
  
  return cv::Point2f(xb + xo, yb + yo);
}

}
