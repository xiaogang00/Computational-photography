#include "FeatureDetection.h"
#include <algorithm>
#include <utility>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ciso646>
#include <opencv2/highgui/highgui.hpp>

#ifndef _MSC_VER
#  define _copysign std::copysign
#endif

namespace vfx {

template<typename T>
inline T bilinear(const cv::Mat& image, float x, float y)
{
  int w = image.size().width, h = image.size().height;
  
  if (x < 0.0f)
    x = 0.0f;
  if (y < 0.0f)
    y = 0.0f;
  
  float x1 = std::min(std::floor(x), float(w-1));
  float y1 = std::min(std::floor(y), float(h-1));
  float x2 = std::min(std::ceil(x), float(w-1));
  float y2 = std::min(std::ceil(y), float(h-1));
  float t;
  
  t = x - x1;
  T c00 = image.at<T>(y1, x1);
  T c01 = image.at<T>(y1, x2);
  T c0 = c00 * (1.0f-t) + c01 * (t);
  
  T c10 = image.at<T>(y2, x1);
  T c11 = image.at<T>(y2, x2);
  T c1 = c10 * (1.0f-t) + c11 * (t);
  
  t = y - y1;
  return c0 * (1.0f-t) + c1 * (t);
}

template<typename T>
inline T bilinear(const cv::Mat& image, const cv::Point2f& pos)
{
  return bilinear<T>(image, pos.x, pos.y);
}

FeatureDetection::FeatureDetection()
{
}

FeatureDetection::~FeatureDetection()
{
}

void FeatureDetection::init(cv::Mat image)
{
  m_SourceImage = image;
  
  cv::Mat greyimage;
  cv::cvtColor(image, greyimage, cv::COLOR_BGR2GRAY);
  
  constructPyramid(greyimage);
  //showPyramid();
}

void FeatureDetection::process()
{
  static const float HARRIS_THRESHOLD = 10.0f;
  
  FeatureList tmpFeatureList;
  
  m_FeatureList.clear();
  m_FeatureList.reserve(4096);
  
  tmpFeatureList.reserve(4096);
  
  for (int lv = 0; lv < m_Pyramid.size(); ++lv) {
    cv::Mat response, features;
    
    tmpFeatureList.clear();
    
    computeHarrisResponse(lv, response);
    markFeatures(response, features, HARRIS_THRESHOLD);
    extractFeatures(lv, features, response, tmpFeatureList);
    refineFeatures(lv, response, tmpFeatureList);
    
    m_FeatureList.insert(m_FeatureList.end(), tmpFeatureList.begin(), tmpFeatureList.end());
  }
  removeNonMaximal(m_FeatureList);
  computeOrientation(m_FeatureList);
  computeDescriptor(m_FeatureList);
}

void FeatureDetection::getFeatures(FeatureInfoList& output)
{
  output.clear();
  output.reserve(m_FeatureList.size());
  for (auto it = m_FeatureList.begin(); it != m_FeatureList.end(); ++it) {
    if (it->valid) {
      FeatureInfo info;
      
      float scale = 1 << it->level;
      info.position = it->position * scale;
      
      std::memcpy(info.descriptor, it->descriptor, sizeof(info.descriptor));
      
      output.push_back(info);
    }
  }
}

void FeatureDetection::showResult(const char* title)
{
  cv::Mat oimage = m_SourceImage.clone();
  int current_level = 0;
  for (const auto& feature : m_FeatureList) {
    if (not feature.valid)
      continue;
    
    if (feature.level > current_level) {
      cv::imshow((title ? title : "features"), oimage);
      cv::waitKey(0);
      oimage = m_SourceImage.clone();
      current_level = feature.level;
    }
    
    draw(oimage, feature);
  }
  cv::imshow((title ? title : "features"), oimage);
  cv::waitKey(0);
}

void FeatureDetection::constructPyramid(cv::Mat greyimage)
{
  static const float SIGMA_P = 1.0f;
  static const int MINSIZE = 64;
  
  m_Pyramid.clear();
  m_Pyramid.reserve(32);
  m_Pyramid.push_back(greyimage);
  
  cv::Size size = greyimage.size();
  int w = size.width /2, h = size.height / 2;
  
  for (; w >= MINSIZE && h >= MINSIZE; w /= 2,h /= 2) {
    cv::Mat src = m_Pyramid.back(), blurred, dst;
    
    cv::GaussianBlur(src, blurred, cv::Size(), SIGMA_P);
    cv::resize(blurred, dst, cv::Size(w, h), 0,0, cv::INTER_NEAREST);
    
    m_Pyramid.push_back(dst);
  }
}

void FeatureDetection::showPyramid()
{
  int level = 0;
  char buf[128];
  for (cv::Mat tmp : m_Pyramid) {
    sprintf(buf, "FeatureDetection::pyramid[%d]", level++);
    cv::imshow(buf, tmp);
  }
  //cv::waitKey(0);
}

void FeatureDetection::computeHarrisResponse(int level, cv::Mat& result)
{
  static const float SIGMA_D = 1.0f;
  static const float SIGMA_I = 1.5f;
  cv::Mat src = m_Pyramid[level];
  
  // Compute Gradients
  cv::Mat gradX, gradY, gradX2, gradY2, gradXY;
  {
    cv::Mat tmp;
    cv::Sobel(src, tmp, CV_32F, 1, 0);
    cv::GaussianBlur(tmp, gradX, cv::Size(), SIGMA_D);
    cv::Sobel(src, tmp, CV_32F, 0, 1);
    cv::GaussianBlur(tmp, gradY, cv::Size(), SIGMA_D);
    cv::multiply(gradX, gradX, tmp);
    cv::GaussianBlur(tmp, gradX2, cv::Size(), SIGMA_I);
    cv::multiply(gradY, gradY, tmp);
    cv::GaussianBlur(tmp, gradY2, cv::Size(), SIGMA_I);
    cv::multiply(gradX, gradY, tmp);
    cv::GaussianBlur(tmp, gradXY, cv::Size(), SIGMA_I);
  }
  if (0) {
    cv::imshow("gradient x", gradX);
    cv::imshow("gradient y", gradY);
    cv::waitKey(0);
  }
  
  // Compute Harris Corner Response
  // M = [ gradX2  gradXY ]
  //     [ gradXY  gradY2 ]
  // det(M) = m00 * m11 - m01 * m10
  // tr(M) = m00 + m11
  //
  cv::Mat_<float> response(src.size());
  int w = response.size().width, h = response.size().height;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      float det, tr;
      det = gradXY.at<float>(y,x);
      det = det * det;
      det = gradX2.at<float>(y,x) * gradY2.at<float>(y,x) - det;
      tr = gradX2.at<float>(y,x) + gradY2.at<float>(y,x);
      if (tr == 0.0f) {
        response.at<float>(y, x) = 0.0f;
      } else {
        response.at<float>(y, x) = det / tr;
      }
    }
  }
  if (0) {
    cv::imshow("response", response);
    cv::waitKey(0);
  }
  
  result = response;
}

void FeatureDetection::markFeatures(cv::Mat response, cv::Mat& result, float threshold)
{
  cv::Mat_<uchar> features(response.size(), 0);
  int w = features.size().width, h = features.size().height;
  
  for (int y = 1; y < h-1; ++y) {
    for (int x = 1; x < w-1; ++x) {
      if (response.at<float>(y,x) > threshold) {
        features.at<uchar>(y,x) = 255;
      } else {
        features.at<uchar>(y,x) = 0;
      }
    }
  }
  
  for (int y = 1; y < h-1; ++y) {
    for (int x = 1; x < w-1; ++x) {
      if (features.at<uchar>(y,x)) {
        bool rejected = false;
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dx = -1; dx <= 1; ++dx) {
            if (dy == 0 || dx == 0)
              continue;
            if (response.at<float>(y+dy,x+dx) > response.at<float>(y,x)) {
              rejected = true;
              break;
            }
          }
          if (rejected)
            break;
        }
        if (rejected)
          features.at<uchar>(y,x) = 0;
      }
    }
  }
  if (0) {
    cv::imshow("features", features);
    cv::waitKey(0);
  }
  
  result = features;
}

void FeatureDetection::extractFeatures(int level, cv::Mat featureMask, cv::Mat response, FeatureList& features)
{
  int w = featureMask.size().width, h = featureMask.size().height;
  
  features.clear();
  
  for (int y = 1; y < h-1; ++y) {
    for (int x = 1; x < w-1; ++x) {
      if (featureMask.at<uchar>(y,x)) {
        FeatureData fdata;
        fdata.position.x = x;
        fdata.position.y = y;
        fdata.level = level;
        fdata.response = response.at<float>(y,x);
        fdata.valid = true;
        
        features.push_back(fdata);
      }
    }
  }
}

void FeatureDetection::refineFeatures(int level, cv::Mat response, FeatureList& features)
{
  int w = response.size().width, h = response.size().height;
  
  cv::Mat f, dfdx, dfdy, d2fdx2, d2fdy2, d2fdxy;
  f = response;
  cv::Sobel(response, dfdx, CV_32F, 1, 0);
  cv::Sobel(response, dfdy, CV_32F, 0, 1);
  cv::Sobel(response, d2fdx2, CV_32F, 2, 0);
  cv::Sobel(response, d2fdy2, CV_32F, 0, 2);
  cv::Sobel(response, d2fdxy, CV_32F, 1, 1);
  
  for (auto it = features.begin(); it != features.end(); ++it) {
    if (not it->valid)
      continue;
    
    float x = it->position.x;
    float y = it->position.y;
    
    cv::Matx<float, 2, 2> matA;
    matA(0, 0) = bilinear<float>(d2fdx2, x, y);
    matA(0, 1) = bilinear<float>(d2fdxy, x, y);
    matA(1, 0) = bilinear<float>(d2fdxy, x, y);
    matA(1, 1) = bilinear<float>(d2fdy2, x, y);
    
    cv::Matx<float, 2, 2> invA = matA.inv(cv::DECOMP_CHOLESKY);
    
    cv::Vec<float, 2> vecB;
    vecB[0] = bilinear<float>(dfdx, x, y);
    vecB[1] = bilinear<float>(dfdy, x, y);
    
    cv::Vec<float, 2> offset = -invA * vecB;
    
    const float MAX_OFFSET = 1.0f;
    if (std::fabs(offset[0]) > MAX_OFFSET)
      offset[0] = _copysign(MAX_OFFSET, offset[0]);
    if (std::fabs(offset[1]) > MAX_OFFSET)
      offset[1] = _copysign(MAX_OFFSET, offset[1]);
    
    //printf("%d %d + %f %f\n", x, y, offset[0], offset[1]);
    //printf("f(%f, %f)=%f -> ", x, y, it->response);
    
    it->position.x += offset[0];
    it->position.y += offset[1];
    
    if (it->position.x < 0.0f)
      it->position.x = 0.0f;
    if (it->position.x >= float(w))
      it->position.x = float(w) - 0.5f;
    if (it->position.y < 0.0f)
      it->position.y = 0.0f;
    if (it->position.y >= float(h))
      it->position.y = float(h) - 0.5f;
    
    // Re-compute Response
    it->response = bilinear<float>(response, it->position);
    
    //printf("f'(%f, %f)=%f\n", it->position.x, it->position.y, it->response);
  }
  
}

void FeatureDetection::removeNonMaximal(FeatureList& features)
{
  int desiredNum = 500;
  int initRadius = 8;
  int step = 1;
  const float NMS_TOLLERATE_RATIO = 0.1f;
  int desiredNumFixed = (int)(desiredNum + NMS_TOLLERATE_RATIO * desiredNum);
  FeatureList::size_type currentNum = desiredNumFixed + 1;
  
  std::vector<bool> valid;
  
  for(int radius = initRadius; currentNum > desiredNumFixed; radius += step) {
    int radiusSquared = radius * radius;
    
    valid.assign(features.size(), true);
    
    for (auto it1 = features.begin(); it1 != features.end(); ++it1) {
      FeatureList::size_type index1 = it1 - features.begin();
      if (not valid[index1] || not it1->valid) {
        valid[index1] = false;
        continue;
      }
      for (auto it2 = features.begin(); it2 != features.end(); ++it2) {
        FeatureList::size_type index2 = it2 - features.begin();
        if (not it2->valid) {
          valid[index2] = false;
          continue;
        }
        
        cv::Point2f delta = it1->position - it2->position;
        float dist2 = delta.dot(delta);
        
        if (index1 == index2 || dist2 >= radiusSquared) {
          continue;
        } else if (it1->response > it2->response) {
          valid[index2] = false;
        } else {
          valid[index1] = false;
          break;
        }
      }
    }
    currentNum = std::count(valid.begin(), valid.end(), true);
  }
  
  FeatureList tmpFeatures;
  tmpFeatures.reserve(features.size());
  for (auto it = features.begin(); it != features.end(); ++it) {
    if (it->valid && valid[it - features.begin()])
      tmpFeatures.push_back(*it);
  }
  features.swap(tmpFeatures);
}

void FeatureDetection::computeOrientation(FeatureList& features)
{
  static const float SIGMA_O = 4.5f;
  
  cv::Mat src, gradX, gradY;
  
  int lv = -1;
  for (auto it = features.begin(); it != features.end(); ++it) {
    if (not it->valid)
      continue;
    
    if (lv != it->level) {
      cv::Mat tmp;
      lv = it->level;
      src = m_Pyramid[lv];
      cv::Sobel(src, tmp, CV_32F, 1, 0);
      cv::GaussianBlur(tmp, gradX, cv::Size(), SIGMA_O);
      cv::Sobel(src, tmp, CV_32F, 0, 1);
      cv::GaussianBlur(tmp, gradY, cv::Size(), SIGMA_O);
    }
    
    int x = it->position.x, y = it->position.y;
    float fx = gradX.at<float>(y,x);
    float fy = gradY.at<float>(y,x);
    float th = std::atan2(fy, fx);
    it->orientation = th;
  }
}

void FeatureDetection::computeDescriptor(FeatureList& features)
{
  //int i, j, u, v;
  //float windows[40][40], new_x, new_y;
  //FILE *fp = fopen("descriptor.txt", "w");
  cv::Mat dimage;
  cv::Matx<float,2,3> xfm;
  cv::Matx<float,3,4> vtx; // array of vertex (x,y,1)
  cv::Matx<float,2,4> vp; // xfm * vtx
  
  for (auto it = features.begin(); it != features.end(); ++it) {
    if (not it->valid)
      continue;
    
    cv::Mat image = m_Pyramid[it->level];
    cv::Size size = image.size();
    
    // transformation from 40x40 window to image space
    xfm = getTransformation(*it);
    
    // check if the region is out of range
    vtx(0,0) = 0;
    vtx(1,0) = 0;
    vtx(2,0) = 1;
    vtx(0,1) = FEATURE_WINDOW_WIDTH;
    vtx(1,1) = 0;
    vtx(2,1) = 1;
    vtx(0,2) = 0;
    vtx(1,2) = FEATURE_WINDOW_WIDTH;
    vtx(2,2) = 1;
    vtx(0,3) = FEATURE_WINDOW_WIDTH;
    vtx(1,3) = FEATURE_WINDOW_WIDTH;
    vtx(2,3) = 1;
    vp = xfm * vtx;
    bool valid = true;
    for (int i = 0; i < 4; ++i) {
      if (vp(0,i) < 0.0f || vp(0,i) >= size.width) {
        valid = false;
        break;
      }
      if (vp(1,i) < 0.0f || vp(1,i) >= size.height) {
        valid = false;
        break;
      }
    }
    if (not valid) {
      it->valid = false;
      continue;
    }
    
    // fill window
    cv::warpAffine(image, dimage, xfm, cv::Size(40, 40), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP);
    
    // save descriptor
    std::memset(it->descriptor, 0, sizeof(float) * 64);
    for (int y = 0; y < 40; ++y) {
      for (int x = 0; x < 40; ++x) {
        int index = (y/5) * 8 + (x/5);
        it->descriptor[index] += dimage.at<uchar>(y,x) / (255.0f * 25.0f);
      }
    }
    
    // normalize
    float mean = 0.0f, stddev = 0.0f;
    for (int i = 0; i < 64; ++i) {
      mean += it->descriptor[i];
    }
    mean /= 64.0f;
    for (int i = 0; i < 64; ++i) {
      float d = it->descriptor[i] - mean;
      stddev += d * d;
    }
    stddev = std::sqrt(stddev / 64.0f);
    for (int i = 0; i < 64; ++i) {
      float x = it->descriptor[i];
      it->descriptor[i] = (x - mean) / stddev;
    }
    
    //cv::imshow("susukino", dimage);
    //cv::waitKey(0);
    /*
    float dx = std::cos(it->orientation);
    float dy = std::sin(it->orientation);
    int img_width = m_Pyramid[it->level].cols;
    int img_height = m_Pyramid[it->level].rows;
    bool flag = true;
    
    if (not it->valid)
      continue;
    
    for (i = -20; i < 20; i++) {
      for (j = -20; j < 20; j++) {
        new_x = i * dx - j * dy + it->position.x;
        new_y = i * dy + j * dx + it->position.y;
        if (new_x < 0 || new_y < 0 || new_x >= img_width-1 || new_y >= img_height-1){
          it->valid = false;
          flag = false;
          break;
        } else {
          windows[i+20][j+20] = bilinear<uchar>(m_Pyramid[it->level], new_x, new_y);
        }
      }
      if(flag == false)
        break;
    }
    
    memset(it->descriptor, 0.0, 64*sizeof(float));
    if (flag == true){
      for (i = 0; i < 40; i++)
        for (j = 0; j < 40; j++)
          it->descriptor[8*(i/5)+(j/5)] += windows[i][j] / 255.0f;
      for (i = 0; i < 64; i++)
        it->descriptor[i] /= 25.0;
      //for (i = 0; i < 64; i++)
      //  fprintf(fp, "%2f\t", features[n].descriptor[i]);
      //fprintf(fp, "\n");
    } else {
      continue;
    }*/
  }
  //printf("%d %d\n", total, temp_count);
  //fclose(fp);
}

cv::Matx<float,2,3> FeatureDetection::getTransformation(const FeatureData& feature)
{
  static const int HALF_WIDTH = FEATURE_WINDOW_WIDTH / 2;
  
  cv::Matx<float,2,3> txfm;
  
  float a = std::cos(feature.orientation), b = std::sin(feature.orientation);
  float cx = HALF_WIDTH, cy = HALF_WIDTH;
  
  txfm(0,0) = a;
  txfm(0,1) = -b;
  txfm(1,0) = b;
  txfm(1,1) = a;
  txfm(0,2) = -cx * (txfm(0,0) + txfm(0,1)) + feature.position.x;
  txfm(1,2) = -cy * (txfm(1,0) + txfm(1,1)) + feature.position.y;
  
  return txfm;
}

void FeatureDetection::draw(cv::Mat canvas, const FeatureData& feature)
{
  static const int HALF_WIDTH = FEATURE_WINDOW_WIDTH / 2;
  /*
  const float orient = feature.orientation;
  const float cosTh = std::cos(orient), sinTh = std::sin(orient);
  const float s = (1 << feature.level);
  const float dx = cosTh * HALF_WIDTH * s, dy = sinTh * HALF_WIDTH * s;
  */
  const float scale = float(1 << feature.level);
  
  cv::Matx<float,2,3> xfm = getTransformation(feature);
  cv::Matx<float,3,6> vec;
  vec(0,0) = HALF_WIDTH;
  vec(1,0) = HALF_WIDTH;
  vec(2,0) = 1;
  vec(0,1) = 0;
  vec(1,1) = 0;
  vec(2,1) = 1;
  vec(0,2) = FEATURE_WINDOW_WIDTH-1;
  vec(1,2) = 0;
  vec(2,2) = 1;
  vec(0,3) = 0;
  vec(1,3) = FEATURE_WINDOW_WIDTH-1;
  vec(2,3) = 1;
  vec(0,4) = FEATURE_WINDOW_WIDTH-1;
  vec(1,4) = FEATURE_WINDOW_WIDTH-1;
  vec(2,4) = 1;
  vec(0,5) = FEATURE_WINDOW_WIDTH-1;
  vec(1,5) = HALF_WIDTH;
  vec(2,5) = 1;
  cv::Matx<float,2,6> transformed = (xfm * vec) * scale;
  
  /*
    
    1---5---2
    |   |   |
    |-- 0 --|
    |   |   |
    3-------4
    
  */
  
  float x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5;
  x0 = transformed(0,0);
  y0 = transformed(1,0);
  x1 = transformed(0,1);
  y1 = transformed(1,1);
  x2 = transformed(0,2);
  y2 = transformed(1,2);
  x3 = transformed(0,3);
  y3 = transformed(1,3);
  x4 = transformed(0,4);
  y4 = transformed(1,4);
  x5 = transformed(0,5);
  y5 = transformed(1,5);
  
  cv::Scalar color(0,0,255);
  
  //x0 = feature.position.x * s, y0 = feature.position.y * s;
  //x5 = x0 + dx;
  //y5 = y0 + dy;
  
  cv::line(canvas, cv::Point(x0,y0), cv::Point(x5,y5), color);
  cv::line(canvas, cv::Point(x1,y1), cv::Point(x2,y2), color);
  cv::line(canvas, cv::Point(x1,y1), cv::Point(x3,y3), color);
  cv::line(canvas, cv::Point(x2,y2), cv::Point(x4,y4), color);
  cv::line(canvas, cv::Point(x3,y3), cv::Point(x4,y4), color);
  /*
  float minX = std::min(std::min(x1, x2), std::min(x3, x4));
  float maxX = std::max(std::max(x1, x2), std::max(x3, x4));
  float minY = std::min(std::min(y1, y2), std::min(y3, y4));
  float maxY = std::max(std::max(y1, y2), std::max(y3, y4));
  cv::Rect roi(cv::Point(minX, minY), cv::Point(maxX, maxY));
  cv::rectangle(canvas, roi, color);
  
  xfm(0,2) -= (roi.x);
  xfm(1,2) -= (roi.y);
  cv::Mat invXfm;
  cv::invertAffineTransform(xfm, invXfm);
  invXfm = invXfm * (1.0f/scale);
  
  cv::Mat dst(canvas, roi);
  
  cv::Mat_<cv::Vec3b> dimage(cv::Size(8, 8));
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 8; ++j) {
      float s = feature.descriptor[i*8+j];
      cv::Vec3b c(255 * s, 255 * s, 255 * s);
      dimage.at<cv::Vec3b>(i,j) = c;
    }
  
  cv::warpAffine(dimage, dst, invXfm, dst.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_TRANSPARENT);
  //cv::imshow("warped", dst);
  */
}

}
