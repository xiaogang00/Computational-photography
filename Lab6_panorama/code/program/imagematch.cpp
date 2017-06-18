#include "FeatureDetection.h"
#include "FeatureMatching.h"
#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include <opencv2/highgui/highgui.hpp>

static std::vector<std::string> s_InputPaths;
static std::vector<cv::Mat> s_InputImages;
static std::vector<std::pair<int,int>> s_ImagePairs;
static std::vector<cv::Matx<float,3,3>> s_Homographies;

static void load_images(const char* listfile);

int main(int argc, char* argv[])
{
  if (argc < 2) {
    std::exit(1);
  }
  
  load_images(argv[1]);
  
  vfx::FeatureMatching matcher;
  
  for (auto it = s_InputPaths.begin(); it != s_InputPaths.end(); ++it) {
    int index = it - s_InputPaths.begin();
    std::cout << "INPUT[" << index << "]: " << *it << std::endl;
    cv::Mat image = cv::imread(it->c_str());
    matcher.addImage(image, s_FocalLengths[index]);
  }
  
  matcher.process();
  matcher.showResult();
  
  matcher.getImages(s_InputImages);
  matcher.getImagePairs(s_ImagePairs);
  matcher.getHomographies(s_Homographies);
  
  for (auto imgpair = s_ImagePairs.begin(); imgpair != s_ImagePairs.end(); ++imgpair) {
    cv::Matx<float,3,3> homo = s_Homographies[imgpair - s_ImagePairs.begin()];
    cv::Matx<float,3,3> invHomo;
    cv::invert(homo, invHomo, cv::DECOMP_SVD);
    
    cv::Mat img1 = s_InputImages[imgpair->first];
    cv::Mat img2 = s_InputImages[imgpair->second];
    
    cv::Mat vtx2(4, 1, CV_32FC2), projVtx2;
    vtx2.at<cv::Point2f>(0, 0) = cv::Point2f(0, 0);
    vtx2.at<cv::Point2f>(1, 0) = cv::Point2f(0, img2.rows);
    vtx2.at<cv::Point2f>(2, 0) = cv::Point2f(img2.cols, img2.rows);
    vtx2.at<cv::Point2f>(3, 0) = cv::Point2f(img2.cols, 0);
    cv::perspectiveTransform(vtx2, projVtx2, invHomo);
    
    cv::Point topLeft2, bottomRight2;
    topLeft2 = bottomRight2 = projVtx2.at<cv::Point2f>(0,0);
    for (int i = 0; i < 4; ++i) {
      cv::Point2f pt = projVtx2.at<cv::Point2f>(i,0);
      topLeft2.x = std::min(topLeft2.x, (int)std::floor(pt.x));
      topLeft2.y = std::min(topLeft2.y, (int)std::floor(pt.y));
      bottomRight2.x = std::max(bottomRight2.x, (int)std::ceil(pt.x));
      bottomRight2.y = std::max(bottomRight2.y, (int)std::ceil(pt.y));
    }
    
    cv::Rect bndRt1(cv::Point(0,0), cv::Point(img1.cols, img1.rows));
    cv::Rect bndRt2(topLeft2, bottomRight2);
    cv::Rect bndCnvs = bndRt1 | bndRt2;
    //printf("%d-%d: %d %d %d %d\n", imgpair->first, imgpair->second, bndCnvs.x, bndCnvs.y, bndCnvs.width, bndCnvs.height);
    
    bndRt1.x -= bndCnvs.x;
    bndRt1.y -= bndCnvs.y;
    bndRt2.x -= bndCnvs.x;
    bndRt2.y -= bndCnvs.y;
    bndCnvs.x = 0, bndCnvs.y = 0;
    
    cv::Matx<float,3,3> matT; // From bndRt2 space to bndRt1 space
    matT(0,0) = 1.0f;
    matT(0,1) = 0.0f;
    matT(0,2) = bndRt2.x - bndRt1.x;
    matT(1,0) = 0.0f;
    matT(1,1) = 1.0f;
    matT(1,2) = bndRt2.y - bndRt1.y;
    matT(2,0) = 0.0f;
    matT(2,1) = 0.0f;
    matT(2,2) = 1.0f;
    
    cv::Mat canvas(bndCnvs.size(), CV_8UC3, cv::Scalar(0,0,0));
    
    img1.copyTo(canvas(bndRt1));
    
    cv::warpPerspective(img2, canvas(bndRt2), homo * matT, bndRt2.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_TRANSPARENT);
    
    for (int i = 0; i < 4; ++i) {
      cv::Point pt1 = projVtx2.at<cv::Point2f>(i, 0);
      cv::Point pt2 = projVtx2.at<cv::Point2f>((i+1)%4, 0);
      pt1 += bndRt1.tl();
      pt2 += bndRt1.tl();
      cv::line(canvas, pt1, pt2, cv::Scalar(0,0,255));
    }
    
    char title[128];
    sprintf(title, "homo-%d-%d", imgpair->first, imgpair->second);
    cv::imshow(title, canvas);
  }
  
  while (cv::waitKey(0) != 27) {
  }
  
  return 0;
}

static void load_images(const char* listfile)
{
  std::string line;
  line.reserve(1024);
  
  std::ifstream stream;
  stream.open(listfile);
  
  while (stream.good()) {
    std::getline(stream, line);
    if (line.length() == 0)
      break;
    //process_line(line);
    s_InputPaths.push_back(line);
    
    line.reserve(1024);
    while (stream.good()) {
      std::getline(stream, line);
      if (line == "...") {
        break;
      }
    }
  }
  stream.close();
}
