#include "FeatureMatching.h"
#include "FeatureDetection.h"
#include "CylindricalPanorama.h"
#include <utility>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <ciso646>
#include <random>
#include <algorithm>
#include <opencv2/highgui/highgui.hpp>
#include <flann/flann.hpp>

namespace vfx {

FeatureMatching::FeatureMatching(TransformationType xfmType)
: m_TransformationType(xfmType)
{
}

FeatureMatching::~FeatureMatching()
{
}

void FeatureMatching::addImage(cv::Mat image, FeatureInfoList&& features)
{
  m_Images.push_back(image);
  m_FeatureLists.emplace_back(std::forward<FeatureInfoList>(features));
}

void FeatureMatching::process()
{
  processMatching();
  constructImagePairs();
  ransac();
  removeInvalidPairs();
}

void FeatureMatching::processMatching()
{
  static const int KNN_SEARCH_NUM = 2;
  static const float FEATURE_MATCHING_THRESHOLD = 0.6f;
  
  std::vector<flann::Index<flann::L2<float>>> trees;
  trees.reserve(m_Images.size());
  
  // Build FLANN Search Trees
  for (auto flist = m_FeatureLists.begin(); flist != m_FeatureLists.end(); ++flist) {
    int index = flist - m_FeatureLists.begin();
    
    flann::Matrix<float> features((float*)((char*)flist->data() + offsetof(FeatureInfo, descriptor)),
                                  flist->size(), 64, sizeof(FeatureInfo));
    
    trees.emplace_back(features, flann::KDTreeIndexParams(4));
    trees.back().buildIndex();
  }
  
  int numImages = m_Images.size();
  
  //std::vector<float> query_data(64, 0.0f);
  std::vector<int> index_data(KNN_SEARCH_NUM, 0);
  std::vector<float> dist_data(KNN_SEARCH_NUM, 0.0f);
  //flann::Matrix<float> query(query_data.data(), 1, 64, 0);
  flann::Matrix<int> index(index_data.data(), 1, KNN_SEARCH_NUM, 0);
  flann::Matrix<float> dist(dist_data.data(), 1, KNN_SEARCH_NUM, 0);
  
  m_BestMatchTables.clear();
  m_BestMatchTables.reserve(numImages);
  
  // Find Match Pairs
  for (int index1 = 0; index1 < numImages; ++index1) {
    FeatureInfoList& flist = m_FeatureLists[index1];
    m_BestMatchTables.emplace_back(flist.size(), numImages, CV_32SC1);
    cv::Mat bmtable = m_BestMatchTables.back();
    
    for (auto f = flist.begin(); f != flist.end(); ++f) {
      int feature_index = f - flist.begin();
      
      //std::memcpy(query.ptr(), f->descriptor, sizeof(float)*64);
      flann::Matrix<float> query(f->descriptor, 1, 64);
      
      for (int index2 = 0; index2 < numImages; ++index2) {
        bmtable.at<int32_t>(feature_index,index2) = -1;
        if (index1 == index2)
          continue;
        trees[index2].knnSearch(query, index, dist, KNN_SEARCH_NUM, flann::SearchParams(128));
        if (dist[0][0] < FEATURE_MATCHING_THRESHOLD * dist[0][1])
          bmtable.at<int32_t>(feature_index, index2) = index[0][0];
      }
    }
  }
  
  // Validate Match Pairs
  for (int index1 = 0; index1 < numImages; ++index1) {
    const FeatureInfoList& flist = m_FeatureLists[index1];
    cv::Mat bmtable = m_BestMatchTables[index1];
    
    for (auto f = flist.begin(); f != flist.end(); ++f) {
      int feature_index = f - flist.begin();
      
      for (int index2 = 0; index2 < numImages; ++index2) {
        if (index1 == index2)
          continue;
        int f2 = bmtable.at<int32_t>(feature_index, index2);
        if (f2 != -1) {
          cv::Mat bmtable2 = m_BestMatchTables[index2];
          if (bmtable2.at<int32_t>(f2, index1) != feature_index)
            bmtable.at<int32_t>(feature_index, index2) = -1;
        }
      }
    }
  }
}

void FeatureMatching::constructImagePairs()
{
  int numImages = m_Images.size();
  
  m_ImageMatches.clear();
  m_ImageMatches.reserve(numImages * (numImages-1) / 2);
  
  for (int index1 = 0; index1 < numImages; ++index1) {
    for (int index2 = index1+1; index2 < numImages; ++index2) {
      int num_features = 0;
      cv::Mat bmtable = m_BestMatchTables[index1];
      for (int i = 0; i < bmtable.rows; ++i) {
        if (bmtable.at<int32_t>(i, index2) != -1)
          ++num_features;
      }
      
      const FeatureInfoList& flist1 = m_FeatureLists[index1];
      const FeatureInfoList& flist2 = m_FeatureLists[index2];
      cv::Mat featpairs(num_features, 2, CV_32FC2);
      for (int i = 0, k = 0; i < bmtable.rows; ++i) {
        int j = bmtable.at<int32_t>(i, index2);
        if (j != -1) {
          featpairs.at<cv::Point2f>(k,0) = flist1[i].position;
          featpairs.at<cv::Point2f>(k,1) = flist2[j].position;
          ++k;
        }
      }
      
      ImagePair match;
      match.imageId1 = index1;
      match.imageId2 = index2;
      match.featurePairs = featpairs;
      match.numInliers = 0;
      m_ImageMatches.emplace_back(std::move(match));
    }
  }
}

void FeatureMatching::ransac()
{
  static const float RANSAC_INLIER_PROBABILITY = 0.6f;
  static const float RANSAC_INLIER_THRESHOLD = 5.0f;
  static const float RANSAC_SUCCESS_PROBABILITY = 0.99f;
  static const int RANSAC_SAMPLE_NUM = 16;
  
  int nIterations = (int)std::ceil(std::log(1.0f-RANSAC_SUCCESS_PROBABILITY) / std::log(1.0 - std::pow(RANSAC_INLIER_PROBABILITY, RANSAC_SAMPLE_NUM)));
  printf("RANSAC ITERATIONS = %d\n", nIterations);
  
  std::random_device rdev;
  std::mt19937 rgen(rdev());
  
  for (ImagePair& match : m_ImageMatches) {
    //printf("Work: Image %d to Image %d:\n", match.imageId1, match.imageId2);
    
    cv::Mat featpairs = match.featurePairs;
    if (featpairs.rows < RANSAC_SAMPLE_NUM * 2)
      continue;
    
    std::uniform_int_distribution<int> distrib(0, featpairs.rows-1);
    
    for (int ransac_count = nIterations; ransac_count > 0; --ransac_count) {
      // Sample Feature Pairs
      int featpair_indices[RANSAC_SAMPLE_NUM];
      std::memset(featpair_indices, -1, sizeof(featpair_indices));
      for (int i = 0, j; i < RANSAC_SAMPLE_NUM; ) {
        featpair_indices[i] = distrib(rgen);
        for (j = 0; j < i; ++j) {
          if (featpair_indices[j] == featpair_indices[i])
            featpair_indices[i] = -1;
        }
        if (featpair_indices[i] != -1)
          ++i;
      }
      //printf("  Sampled:");
      //for (int i = 0; i < RANSAC_SAMPLE_NUM; ++i)
      //  printf(" %d", featpair_indices[i]);
      //printf("\n");
      
      // Compute Homography
      cv::Matx<float,3,3> homo;
      {
        // Extract sampled feature pairs
        cv::Point2f pt1[RANSAC_SAMPLE_NUM];
        cv::Point2f pt2[RANSAC_SAMPLE_NUM];
        for (int i = 0; i < RANSAC_SAMPLE_NUM; ++i) {
          pt1[i] = featpairs.at<cv::Point2f>(featpair_indices[i], 0);
          pt2[i] = featpairs.at<cv::Point2f>(featpair_indices[i], 1);
        }
        
        switch (m_TransformationType) {
        case TransformationType::HOMOGRAPHY:
          computeHomography(RANSAC_SAMPLE_NUM, pt1, pt2, homo);
          break;
        case TransformationType::AFFINE:
          computeAffine(RANSAC_SAMPLE_NUM, pt1, pt2, homo);
          break;
        case TransformationType::TRANSLATION:
          computeTranslation(RANSAC_SAMPLE_NUM, pt1, pt2, homo);
          break;
        }
      }
      
      // Project Points in Image 1 to Image 2 with the Homography
      cv::Mat pos1 = featpairs(cv::Rect(0,0,1,featpairs.rows));
      cv::Mat pos2 = featpairs(cv::Rect(1,0,1,featpairs.rows));
      cv::Mat proj_pos2;
      cv::perspectiveTransform(pos1, proj_pos2, homo);
      CV_Assert(proj_pos2.type() == CV_32FC2);
      
      // Analysis
      int numInliers = 0;
      for (int i = 0; i < proj_pos2.rows; ++i) {
        //cv::Point2f p1 = pos1.at<cv::Point2f>(i,0);
        cv::Point2f p2 = proj_pos2.at<cv::Point2f>(i,0);
        cv::Point2f p2_exp = pos2.at<cv::Point2f>(i,0);
        cv::Point2f d = p2 - p2_exp;
        float dist = d.dot(d);
        if (dist < RANSAC_INLIER_THRESHOLD) {
          ++numInliers;
        }
      }
      //printf("  Inliers = %d, %f %%\n", numInliers, float(numInliers)/float(featpairs.rows)*100.0f);
      
      if (numInliers > match.numInliers) {
        match.numInliers = numInliers;
        match.inliers.create(featpairs.rows, 1, CV_8UC1);
        for (int i = 0; i < proj_pos2.rows; ++i) {
          //cv::Point2f p1 = pos1.at<cv::Point2f>(i,0);
          cv::Point2f p2 = proj_pos2.at<cv::Point2f>(i,0);
          cv::Point2f p2_exp = pos2.at<cv::Point2f>(i,0);
          cv::Point2f d = p2 - p2_exp;
          float dist = d.dot(d);
          if (dist < RANSAC_INLIER_THRESHOLD) {
            match.inliers.at<uchar>(i,0) = 0xF;
          } else {
            match.inliers.at<uchar>(i,0) = 0;
          }
        }
        for (int i = 0; i < RANSAC_SAMPLE_NUM; ++i) {
          match.inliers.at<uchar>(featpair_indices[i], 0) |= 0x80;
        }
        match.homography = homo;
      }
    }
  }
}

void FeatureMatching::removeInvalidPairs()
{
  std::vector<ImagePair> tmpList;
  tmpList.reserve(m_ImageMatches.size());
  
  for (ImagePair& match : m_ImageMatches) {
    if (match.valid()) {
      tmpList.emplace_back(std::move(match));
    }
  }
  
  m_ImageMatches.swap(tmpList);
}

void FeatureMatching::showResult()
{
  char title[128];
  
  for (ImagePair& match : m_ImageMatches) {
    int index1 = match.imageId1;
    int index2 = match.imageId2;
    std::sprintf(title, "ransac-%d-%d.jpg", index1, index2);
    
    if (match.valid()) {
      cv::Mat canvas = drawImagePair(match);
      cv::imshow(title, canvas);
      
      printf("%s 's Homography = \n", title);
      cv::Matx<float,3,3>& h = match.homography;
      for (int i = 0; i < 3; ++i) {
        printf("    [ ");
        for (int j = 0; j < 3; ++j)
          printf("%f ", h(i,j));
        printf("]\n");
      }
      
      cv::imwrite(g_RansacOutputDirectory + title, canvas);
    }
  }
}


void FeatureMatching::getImages(std::vector<cv::Mat>& output)
{
  output.clear();
  output.reserve(m_Images.size());
  for (cv::Mat img : m_Images) {
    output.push_back(img);
  }
}

void FeatureMatching::getImagePairs(std::vector<std::pair<int,int>>& output)
{
  output.clear();
  output.reserve(m_ImageMatches.size());
  
  for (ImagePair& match : m_ImageMatches) {
    if (match.valid())
      output.push_back(std::make_pair(match.imageId1, match.imageId2));
  }
}

void FeatureMatching::getHomographies(std::vector<cv::Matx<float,3,3>>& output)
{
  output.clear();
  output.reserve(m_ImageMatches.size());
  
  for (ImagePair& match : m_ImageMatches) {
    if (match.valid())
      output.push_back(match.homography);
  }
}

cv::Mat FeatureMatching::drawImagePair(int index1, int index2, bool drawline)
{
  cv::Mat src1 = m_Images[index1];
  const FeatureInfoList& flist1 = m_FeatureLists[index1];
  cv::Mat bmtable1 = m_BestMatchTables[index1];
  
  cv::Mat src2 = m_Images[index2];
  const FeatureInfoList& flist2 = m_FeatureLists[index2];
  cv::Mat bmtable2 = m_BestMatchTables[index2];
  
  // create output canvas
  cv::Size canvas_size;
  canvas_size.width = std::max(src1.cols, src2.cols);
  canvas_size.height = (src1.rows + src2.rows);
  cv::Mat canvas(canvas_size, CV_8UC3);
  
  //
  cv::Point offset2(0, src1.rows);
  
  // copy images
  {
    cv::Rect roi1(cv::Point(0,0), src1.size());
    cv::Rect roi2(cv::Point(0,roi1.height), src2.size());
    src1.copyTo(canvas(roi1));
    src2.copyTo(canvas(roi2));
  }
  
  // draw features
  {
    for (auto f = flist1.begin(); f != flist1.end(); ++f) {
      cv::Point pos(f->position.x, f->position.y);
      cv::Scalar color;
      if (bmtable1.at<int32_t>(f-flist1.begin(), index2) != -1) {
        color = cv::Scalar(0,255,0);
      } else {
        color = cv::Scalar(0,0,255);
      }
      cv::line(canvas, cv::Point(pos.x-5,pos.y), cv::Point(pos.x+5,pos.y), color);
      cv::line(canvas, cv::Point(pos.x,pos.y-5), cv::Point(pos.x,pos.y+5), color);
    }
    for (auto f = flist2.begin(); f != flist2.end(); ++f) {
      cv::Point pos = cv::Point(f->position) + offset2;
      
      cv::Scalar color;
      if (bmtable2.at<int32_t>(f-flist2.begin(), index1) != -1) {
        color = cv::Scalar(0,255,0);
      } else {
        color = cv::Scalar(0,0,255);
      }
      
      cv::line(canvas, cv::Point(pos.x-5,pos.y), cv::Point(pos.x+5,pos.y), color);
      cv::line(canvas, cv::Point(pos.x,pos.y-5), cv::Point(pos.x,pos.y+5), color);
    }
  }
  
  // draw lines
  if (drawline) {
    for (auto f = flist1.begin(); f != flist1.end(); ++f) {
      int f1 = f-flist1.begin();
      int f2 = bmtable1.at<int32_t>(f1, index2);
      if (f2 == -1)
        continue;
      cv::Point pos1 = flist1[f1].position;
      cv::Point pos2 = cv::Point(flist2[f2].position) + offset2;
      cv::Scalar color(255,0,0);
      cv::line(canvas, pos1, pos2, color);
    }
  }
  
  return canvas;
}

cv::Mat FeatureMatching::drawImagePair(ImagePair& pair)
{
  int index1 = pair.imageId1;
  int index2 = pair.imageId2;
  cv::Mat canvas = drawImagePair(index1, index2, false);
  cv::Point offset2(0, m_Images[index1].rows);
  
  cv::Mat featpairs = pair.featurePairs;
  for (int i = 0; i < featpairs.rows; ++i) {
    cv::Point pos1(featpairs.at<cv::Point2f>(i,0));
    cv::Point pos2(featpairs.at<cv::Point2f>(i,1));
    
    cv::Scalar color(128,0,0);
    if (pair.inliers.rows > 0) {
      uchar flag = pair.inliers.at<uchar>(i,0);
      if (flag & 0x0F)
        color = cv::Scalar(0,200,0);
      if (flag & 0x80) {
        color[2] = 255;
        color[1] = 255;
      }
    }
    
    cv::line(canvas, pos1, pos2 + offset2, color);
  }
  
  return canvas;
}

void FeatureMatching::computeHomography(int n, cv::Point2f* p1, cv::Point2f* p2, cv::OutputArray homo)
{
  homo.create(3, 3, CV_32FC1);
  cv::Mat dst = homo.getMat();
  
  cv::Mat matA(2*n, 9, CV_32FC1);
  cv::Mat matX(9, 1, CV_32FC1);
  
  for (int i = 0; i < n; ++i) {
    int ii = 2 * i;
    matA.at<float>(ii, 0) = p1[i].x;
    matA.at<float>(ii, 1) = p1[i].y;
    matA.at<float>(ii, 2) = 1.0f;
    matA.at<float>(ii, 3) = 0.0f;
    matA.at<float>(ii, 4) = 0.0f;
    matA.at<float>(ii, 5) = 0.0f;
    matA.at<float>(ii, 6) = -p2[i].x * p1[i].x;
    matA.at<float>(ii, 7) = -p2[i].x * p1[i].y;
    matA.at<float>(ii, 8) = -p2[i].x;
    ii += 1;
    matA.at<float>(ii, 0) = 0.0f;
    matA.at<float>(ii, 1) = 0.0f;
    matA.at<float>(ii, 2) = 0.0f;
    matA.at<float>(ii, 3) = p1[i].x;
    matA.at<float>(ii, 4) = p1[i].y;
    matA.at<float>(ii, 5) = 1.0f;
    matA.at<float>(ii, 6) = -p2[i].y * p1[i].x;
    matA.at<float>(ii, 7) = -p2[i].y * p1[i].y;
    matA.at<float>(ii, 8) = -p2[i].y;
  }
  
  cv::SVD::solveZ(matA, matX);
  
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      dst.at<float>(i, j) = matX.at<float>(i*3+j, 0);
    }
  }
}

void FeatureMatching::computeAffine(int n, cv::Point2f* p1, cv::Point2f* p2, cv::OutputArray homo)
{
  homo.create(3, 3, CV_32FC1);
  cv::Mat dst = homo.getMat();
  
  cv::Mat matA(2*n, 6, CV_32FC1);
  cv::Mat matX(6, 1, CV_32FC1);
  cv::Mat matB(2*n, 1, CV_32FC1);
  
  for (int i = 0; i < n; ++i) {
    int ii = 2 * i;
    matA.at<float>(ii, 0) = p1[i].x;
    matA.at<float>(ii, 1) = p1[i].y;
    matA.at<float>(ii, 2) = 1.0f;
    matA.at<float>(ii, 3) = 0.0f;
    matA.at<float>(ii, 4) = 0.0f;
    matA.at<float>(ii, 5) = 0.0f;
    matB.at<float>(ii, 0) = p2[i].x;
    ii += 1;
    matA.at<float>(ii, 0) = 0.0f;
    matA.at<float>(ii, 1) = 0.0f;
    matA.at<float>(ii, 2) = 0.0f;
    matA.at<float>(ii, 3) = p1[i].x;
    matA.at<float>(ii, 4) = p1[i].y;
    matA.at<float>(ii, 5) = 1.0f;
    matB.at<float>(ii, 0) = p2[i].y;
  }
  
  cv::solve(matA, matB, matX, cv::DECOMP_SVD);
  
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      dst.at<float>(i, j) = matX.at<float>(i*3+j, 0);
    }
  }
  dst.at<float>(2, 0) = 0.0f;
  dst.at<float>(2, 1) = 0.0f;
  dst.at<float>(2, 2) = 1.0f;
}

void FeatureMatching::computeTranslation(int n, cv::Point2f* p1, cv::Point2f* p2, cv::OutputArray homo)
{
  homo.create(3, 3, CV_32FC1);
  cv::Mat dst = homo.getMat();
  
  float tx = 0, ty = 0;
  for (int i = 0; i < n; ++i) {
    tx += p2[i].x - p1[i].x;
    ty += p2[i].y - p1[i].y;
  }
  tx /= n, ty /= n;
  
  dst.at<float>(0, 0) = 1.0f;
  dst.at<float>(0, 1) = 0.0f;
  dst.at<float>(0, 2) = tx;
  dst.at<float>(1, 0) = 0.0f;
  dst.at<float>(1, 1) = 1.0f;
  dst.at<float>(1, 2) = ty;
  dst.at<float>(2, 0) = 0.0f;
  dst.at<float>(2, 1) = 0.0f;
  dst.at<float>(2, 2) = 1.0f;
}

}
