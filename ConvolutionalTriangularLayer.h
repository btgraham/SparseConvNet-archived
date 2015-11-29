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
  int minActiveInputs;
  ConvolutionalTriangularLayer(cudaMemStream &memStream, int filterSize,
                               int filterStride, int dimension, int nFeaturesIn,
                               int minActiveInputs);
  void preprocess(SpatiallySparseBatch &batch,
                  SpatiallySparseBatchInterface &input,
                  SpatiallySparseBatchInterface &output);
  void forwards(SpatiallySparseBatch &batch,
                SpatiallySparseBatchInterface &input,
                SpatiallySparseBatchInterface &output);
  void backwards(SpatiallySparseBatch &batch,
                 SpatiallySparseBatchInterface &input,
                 SpatiallySparseBatchInterface &output, float learningRate,
                 float momentum);
  int calculateInputSpatialSize(int outputSpatialSize);
};
