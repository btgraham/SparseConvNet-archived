#pragma once
#include "SpatiallySparseLayer.h"

class ConvolutionalTriangularLayer : public SpatiallySparseLayer {
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
  ConvolutionalTriangularLayer(int filterSize,
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
