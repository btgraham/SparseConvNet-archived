#pragma once
#include "readImageToMat.h"
#include "Picture.h"
#include <vector>

// If filename contain an image filename, then it gets loaded as needed;
// otherwise mat is used to stores the image

class OpenCVPicture : public Picture {
  float scaleUCharColor(float col);

public:
  std::string filename;
  int xOffset; // Shift to the right
  int yOffset; // Shift down
  int backgroundColor;
  float scale2;
  float scale2xx, scale2xy, scale2yy;
  cv::Mat mat;
  // to hold "filename" image file in memory if RAM allows
  std::vector<char> rawData;

  OpenCVPicture(int xSize, int ySize, int nInputFeatures,
                unsigned char backgroundColor, int label_ = -1);
  OpenCVPicture(std::string filename, unsigned char backgroundColor = 128,
                int label_ = -1);
  ~OpenCVPicture();
  Picture *distort(RNG &rng, batchType type = TRAINBATCH);
  void affineTransform(float c00, float c01, float c10, float c11);
  void codifyInputData(SparseGrid &grid, std::vector<float> &features,
                       int &nSpatialSites, int spatialSize);
  void jiggle(RNG &rng, int offlineJiggle);
  void jiggleFit(RNG &rng, int subsetSize, float minFill = -1);
  void elasticDistortion(RNG &rng, float amplitude = 30, float radius = 30);
  void colorDistortion(RNG &rng, int sigma1, int sigma2, int sigma3,
                       int sigma4);
  void loadData(int scale, int flags = 1);
  void loadDataWithoutScaling(int flags = 1);
  void loadDataWithoutScalingRemoveModalColor(int flags = 1);
  void loadDataWithoutScalingRemoveMeanColor(int flags = 1);
  void randomCrop(RNG &rng, int subsetSize);
  void blur(float radius);
  void addSpatiallyCoherentNoise(RNG &rng, float amplitude, float radius);
  void multiplySpatiallyCoherentNoise(RNG &rng, float amplitude, float radius);
  void centerMass();
  int area(); // Function to compute number of non-background pixels = area of
              // object
  std::string identify();
};

void matrixMul2x2inPlace(float &c00, float &c01, float &c10, float &c11,
                         float a00, float a01, float a10, float a11);
