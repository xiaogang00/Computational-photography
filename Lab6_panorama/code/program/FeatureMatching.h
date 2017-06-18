#pragma once

#include "FeatureDetection.h"

namespace vfx {

enum class TransformationType {
  HOMOGRAPHY,
  AFFINE,
  TRANSLATION,
};

class FeatureMatching {
public:
  FeatureMatching(TransformationType xfmType);
  ~FeatureMatching();
  
  void addImage(cv::Mat image, FeatureInfoList&& features);
  void process();
  void showResult();
  
  void getImages(std::vector<cv::Mat>& output);
  void getImagePairs(std::vector<std::pair<int,int>>& output);
  void getHomographies(std::vector<cv::Matx<float,3,3>>& output);
  
private:
  struct ImagePair {
    int imageId1;
    int imageId2;
    cv::Matx<float, 3, 3> homography;
    cv::Mat featurePairs; // at<Point2f>(i, 0) -> at<Point2f>(i, 1)
    cv::Mat inliers; // m[i,0] == 1 => is inlier
    int numInliers;
    
    inline bool valid() const {
      return ((numInliers - 5.9f) > (0.22f * featurePairs.rows));
    }
  };
  
private:
  void processMatching();
  void constructImagePairs();
  void ransac();
  void removeInvalidPairs();
  cv::Mat drawImagePair(int index1, int index2, bool drawline = true);
  cv::Mat drawImagePair(ImagePair& pair);
  void computeHomography(int n, cv::Point2f* p1, cv::Point2f* p2, cv::OutputArray homo);
  void computeAffine(int n, cv::Point2f* p1, cv::Point2f* p2, cv::OutputArray homo);
  void computeTranslation(int n, cv::Point2f* p1, cv::Point2f* p2, cv::OutputArray homo);
  
private:
  std::vector<FeatureInfoList> m_FeatureLists;
  std::vector<cv::Mat> m_Images;
  
  // [image-id].at<int>(feature-id, image-id), -1 if no match
  std::vector<cv::Mat> m_BestMatchTables;
  
  std::vector<ImagePair> m_ImageMatches;
  
  // Options
  TransformationType m_TransformationType;
  
};

}
