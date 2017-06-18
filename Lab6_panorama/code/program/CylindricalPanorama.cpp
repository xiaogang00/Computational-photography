#include "CylindricalPanorama.h"
#include "FeatureDetection.h"
#include "FeatureMatching.h"
#include "ImageBlending.h"
#include "ImageWarping.h"
#include "DisjointSet.h"
#include <opencv2/highgui/highgui.hpp>
#include <algorithm>

namespace vfx {

CylindricalPanorama::CylindricalPanorama()
{
}

CylindricalPanorama::~CylindricalPanorama()
{
  
}

void CylindricalPanorama::addImage(cv::Mat image, float focal)
{
  m_InputImages.push_back(image);
  m_FocalLengths.push_back(focal);
}

void CylindricalPanorama::process()
{
  processFeatureMatching();
  buildImageGraph();
  for (auto& pano : m_Panoramas) {
    pano.process();
  }
  printf("DONE\n");
}

void CylindricalPanorama::processFeatureMatching()
{
  FeatureMatching matcher(g_TransformationType);
  
  for (auto it = m_InputImages.begin(); it != m_InputImages.end(); ++it) {
    int index = it - m_InputImages.begin();
    cv::Mat& image = *it;
    float focal = m_FocalLengths[index];
    
    // Get Features
    FeatureInfoList features;
    {
      printf("Image[%d]: Feature Detection\n", index);
      FeatureDetection detector;
      detector.init(image);
      detector.process();
      detector.getFeatures(features);
    }
    printf("Image[%d]: %d features detected.\n", index, (int)features.size());
    
    // Warp Features and Image
    if (g_EnableWarping) {
      cv::Mat warppedImage;
      {
        printf("Image[%d]: Warpping\n", index);
        ImageWarping warp(focal, image.size());
        for (FeatureInfo& feat : features) {
          feat.position = warp(feat.position);
        }
        warppedImage = warp(image);
      }
      matcher.addImage(warppedImage, std::move(features));
    } else {
      matcher.addImage(image, std::move(features));
    }
  }
  
  printf("Process Matching\n");
  matcher.process();
  if (g_ShowRansacResult)
    matcher.showResult();
  
  //matcher.getImages(s_InputImages);
  matcher.getImagePairs(m_ImagePairs);
  matcher.getHomographies(m_Homographies);
}

void CylindricalPanorama::buildImageGraph()
{
  printf("Build Image Graph\n");
  
  int num = m_InputImages.size();
  
  // Define Vertices
  for (int i = 0; i < num; ++i) {
    int v = m_ImageGraph.addVertex();
    CV_Assert(v == i);
  }
  
  // Define Edges
  for (auto imgpair = m_ImagePairs.begin(); imgpair != m_ImagePairs.end(); ++imgpair) {
    cv::Matx<float,3,3> homo = m_Homographies[imgpair - m_ImagePairs.begin()];
    cv::Matx<float,3,3> invHomo;
    cv::invert(homo, invHomo, cv::DECOMP_SVD);
    
    int v1 = imgpair->first, v2 = imgpair->second;
    m_ImageGraph.addEdge(v1, v2, homo);
    m_ImageGraph.addEdge(v2, v1, invHomo);
  }
  
  // Initialize Connected Component
  DisjointSet componentset(num);
  
  // Merge Components
  for (const std::pair<int,int>& imgpair : m_ImagePairs) {
    componentset.merge(imgpair.first, imgpair.second);
  }
  
  // Show how many components
  std::vector<int> roots(num);
  for (int i = 0; i < num; ++i) {
    roots[i] = componentset.find(i);
  }
  std::sort(roots.begin(), roots.end());
  roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
  
  printf("Components: [");
  for (int k : roots) {
    printf(" %d", k);
  }
  printf(" ]\n");
  
  m_Panoramas.clear();
  m_Panoramas.reserve(roots.size());
  
  std::vector<cv::Matx<float,3,3>> transforms;
  transforms.resize(num);
  
  for (int root : roots) {
    std::vector<cv::Mat> images;
    std::vector<float> focals;
    std::vector<cv::Matx<float,3,3>> local_transforms;
    
    
    m_ImageGraph.dfs(root,
    [root, &transforms, &focals, &images, &local_transforms, this]
    (int node, int parent, const cv::Matx<float,3,3>* edgeMat)
    {
      if (parent == -1) {
        cv::setIdentity(transforms[node]);
      } else {
        transforms[node] = (*edgeMat) * transforms[parent];
      }
      
      images.push_back(m_InputImages[node]);
      focals.push_back(m_FocalLengths[node]);
      local_transforms.push_back(transforms[node]);
      
      printf("dfs[%d]: %d -> %d\n", root, parent, node);
      
      cv::Matx<float,3,3>& h = transforms[node];
      for (int i = 0; i < 3; ++i) {
        printf("    [ ");
        for (int j = 0; j < 3; ++j)
          printf("%f ", h(i,j));
        printf("]\n");
      }
    });
    
    m_Panoramas.emplace_back(std::move(images), std::move(focals), std::move(local_transforms));
  }
}

void CylindricalPanorama::getResult(std::vector<cv::Mat>& output)
{
  output.clear();
  output.reserve(m_Panoramas.size());
  for (auto& pano : m_Panoramas) {
    cv::Mat img;
    pano.getResult(img);
    output.push_back(img);
  }
}

TransformationType g_TransformationType = TransformationType::AFFINE;
bool g_EnableWarping = true;
bool g_DrawBoundingRects = false;
bool g_ShowRansacResult = true;
std::string g_OutputDirectory;
std::string g_RansacOutputDirectory;

}
