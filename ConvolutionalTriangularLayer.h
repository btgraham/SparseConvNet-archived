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
  int lPad;
  int rPad;
  ConvolutionalTriangularLayer(int filterSize,
                               int filterStride,
                               int dimension,
                               int nFeaturesIn,
                               int lPad=0,
                               int rPad=0);
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
