#pragma once
#include "SpatiallySparseLayer.h"

class ConvolutionalLayer : public SpatiallySparseLayer {
private:
  int fs;
public:
  int inSpatialSize;
  int outSpatialSize;
  int filterSize;
  int filterStride;
  int dimension;
  int nFeaturesIn;
  int nFeaturesOut;
  ConvolutionalLayer(int filterSize,
                     int filterStride,
                     int dimension,
                     int nFeaturesIn);
  void preprocess
  (SpatiallySparseBatch &batch,
   SpatiallySparseBatchInterface &input,
   SpatiallySparseBatchInterface &output);
  void forwards
  (SpatiallySparseBatch &batch,
   SpatiallySparseBatchInterface &input,
   SpatiallySparseBatchInterface &output);
  void backwards(SpatiallySparseBatch &batch,
                 SpatiallySparseBatchInterface &input,
                 SpatiallySparseBatchInterface &output,
                 float learningRate);
  int calculateInputSpatialSize(int outputSpatialSize);
};

template <typename t> void convolutionFeaturesPresent(std::vector<t>& d_src, std::vector<t>& d_dest, int nf, int nfp, int nCopies);
void propForwardToMatrixMultiply(float* inFeatures, float* outFeatures, int* rules, int count, int nIn);
void propBackwardFromMatrixMultiply(float* inDFeatures, float* outDFeatures, int* rules, int count, int nIn);
