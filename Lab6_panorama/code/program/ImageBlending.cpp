#include "ImageBlending.h"
#include "ImageWarping.h"
#include <opencv2/highgui/highgui.hpp>
#include "CylindricalPanorama.h"

namespace vfx {

ImageBlending::ImageBlending(std::vector<cv::Mat>&& images, std::vector<float>&& focal, std::vector<cv::Matx<float,3,3>>&& transforms)
: m_Images(std::forward<std::vector<cv::Mat>>(images)),
  m_FocalLengths(std::forward<std::vector<float>>(focal)),
  m_Transforms(std::forward<std::vector<cv::Matx<float,3,3>>>(transforms))
{
  std::vector<cv::Matx<float,3,3>> invTransforms;
  invTransforms.reserve(m_Transforms.size());
  
  for (cv::Matx<float,3,3>& mat : m_Transforms) {
    invTransforms.emplace_back();
    cv::invert(mat, invTransforms.back(), cv::DECOMP_SVD);
  }
  
  m_Bounds.reserve(m_Images.size());
  for (auto img = m_Images.begin(); img != m_Images.end(); ++img) {
    int index = img - m_Images.begin();
    cv::Matx<float,3,3>& mat = invTransforms[index];
    cv::Rect origRect(cv::Point(0,0), img->size());
    
    cv::Mat vertices(4, 1, CV_32FC2), verticesProj;
    vertices.at<cv::Point2f>(0, 0) = cv::Point2f(0, 0);
    vertices.at<cv::Point2f>(1, 0) = cv::Point2f(0, img->rows);
    vertices.at<cv::Point2f>(2, 0) = cv::Point2f(img->cols, img->rows);
    vertices.at<cv::Point2f>(3, 0) = cv::Point2f(img->cols, 0);
    cv::perspectiveTransform(vertices, verticesProj, mat);
    
    cv::Point topLeft, bottomRight;
    topLeft = bottomRight = verticesProj.at<cv::Point2f>(0,0);
    for (int i = 0; i < 4; ++i) {
      cv::Point2f pt = verticesProj.at<cv::Point2f>(i,0);
      topLeft.x = std::min(topLeft.x, (int)std::floor(pt.x));
      topLeft.y = std::min(topLeft.y, (int)std::floor(pt.y));
      bottomRight.x = std::max(bottomRight.x, (int)std::ceil(pt.x));
      bottomRight.y = std::max(bottomRight.y, (int)std::ceil(pt.y));
    }
    
    TransformedRect rect;
    rect.bound = cv::Rect(topLeft, bottomRight);
    for (int i = 0; i < 4; ++i)
      rect.vertex[i] = verticesProj.at<cv::Point2f>(i,0);
    m_Bounds.push_back(rect);
  }
  
  cv::Rect canvasBound = m_Bounds[0].bound;
  for (auto& rt : m_Bounds) {
    canvasBound = canvasBound | rt.bound;
  }
  printf("canvas: %d %d %d %d\n", canvasBound.x, canvasBound.y, canvasBound.width, canvasBound.height);
  
  for (auto rt = m_Bounds.begin(); rt != m_Bounds.end(); ++rt) {
    rt->bound.x -= canvasBound.x;
    rt->bound.y -= canvasBound.y;
    for (int i = 0; i < 4; ++i)
      rt->vertex[i] = rt->vertex[i] - cv::Point2f(canvasBound.tl());
    
    cv::Matx<float,3,3>& transform = m_Transforms[rt-m_Bounds.begin()];
    cv::Matx<float,3,3> trans;
    cv::setIdentity(trans);
    trans(0,2) += canvasBound.x;
    trans(1,2) += canvasBound.y;
    transform = transform * trans;
    
    //cv::Matx<float,3,3>& h = trans;
    //for (int i = 0; i < 3; ++i) {
    //  printf("    [ ");
    //  for (int j = 0; j < 3; ++j)
    //    printf("%f ", h(i,j));
    //  printf("]\n");
    //}
  }
  canvasBound.x = 0;
  canvasBound.y = 0;
  framebuffer.create(canvasBound.size(), CV_8UC3);
  stencilbuffer.create(canvasBound.size(), CV_8UC1);
  
}

ImageBlending::ImageBlending(ImageBlending&& other)
#if defined(_MSC_VER) && (_MSC_VER >= 1800)
: m_Images(std::move(other.m_Images)),
  m_FocalLengths(std::move(other.m_FocalLengths)),
  m_Transforms(std::move(other.m_Transforms))
#else
: ImageBlending(std::move(other.m_Images),
                std::move(other.m_FocalLengths),
                std::move(other.m_Transforms))
#endif
{
}

void ImageBlending::process()
{
  framebuffer.setTo(cv::Scalar(0,0,0));
  stencilbuffer.setTo(cv::Scalar(0,0,0));
  
  bool enableWarping = g_EnableWarping;
  
  for (auto rt = m_Bounds.begin(); rt != m_Bounds.end(); ++rt) {
    int index = rt - m_Bounds.begin();
    
    cv::Matx<float,3,3> proj = m_Transforms[index];
    cv::Mat image = m_Images[index];
    float focal = m_FocalLengths[index];
    ImageWarping warp(focal, image.size());
    cv::Mat output = framebuffer(rt->bound); // ROI
    cv::Mat stencil = stencilbuffer(rt->bound); // ROI
    
    for (int yw = 0; yw < output.rows; ++yw) {
      for (int xw = 0; xw < output.cols; ++xw) {
        // Transform to Texture Space
        cv::Vec<float,3> pos1(xw + rt->bound.x, yw + rt->bound.y, 1.0f);
        cv::Vec<float,3> pos2 = proj * pos1;
        cv::Point2f srcpos = cv::Point2f(pos2[0]/pos2[2], pos2[1]/pos2[2]);
        
        if (enableWarping)
          srcpos = warp.project(srcpos);
        
        int x0, y0, x1, y1;
        x0 = std::floor(srcpos.x);
        y0 = std::floor(srcpos.y);
        x1 = std::ceil(srcpos.x);
        y1 = std::ceil(srcpos.y);
        
        if ((x0 >= 0 && x1 < image.cols) && (y0 >= 0 && y1 < image.rows)) {
          float tx = srcpos.x - (float)x0;
          float ty = srcpos.y - (float)y0;
          cv::Vec3f c0 = image.at<cv::Vec3b>(y0,x0);
          cv::Vec3f c1 = image.at<cv::Vec3b>(y0,x1);
          cv::Vec3f c2 = image.at<cv::Vec3b>(y1,x0);
          cv::Vec3f c3 = image.at<cv::Vec3b>(y1,x1);
          cv::Vec3f c_1 = c0 + (c1-c0) * tx;
          cv::Vec3f c_2 = c2 + (c3-c2) * tx;
          cv::Vec3f c_3 = c_1 + (c_2-c_1) * ty;
          
          int oldweight = stencil.at<uchar>(yw,xw);
          cv::Vec3f oldcolor = output.at<cv::Vec3b>(yw,xw);
          cv::Vec3f newcolor = (oldcolor * oldweight + c_3 ) / (oldweight+1);
          output.at<cv::Vec3b>(yw,xw) = newcolor;
          stencil.at<uchar>(yw,xw) = oldweight + 1;
        }
      }
    }
  }
  
  if (g_DrawBoundingRects) {
    for (auto rt = m_Bounds.begin(); rt != m_Bounds.end(); ++rt) {
      for (int i = 0; i < 4; ++i) {
        cv::Point p1 = rt->vertex[i];
        cv::Point p2 = rt->vertex[(i+1)%4];
        cv::line(framebuffer, p1, p2, cv::Scalar(0,0,255));
      }
      cv::rectangle(framebuffer, rt->bound, cv::Scalar(255,0,0));
    }
  }
  
  //cv::imshow("panorama", framebuffer);
}

void ImageBlending::getResult(cv::OutputArray output)
{
  output.create(framebuffer.size(), CV_8UC3);
  cv::Mat dst = output.getMat();
  framebuffer.copyTo(dst);
}

}

