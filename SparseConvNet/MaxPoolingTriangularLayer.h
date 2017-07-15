#pragma once
#include <iostream>
#include <vector>
#include "SpatiallySparseLayer.h"
#include "Rng.h"

class MaxPoolingTriangularLayer : public SpatiallySparseLayer {
public:
  int inSpatialSize;
  int outSpatialSize;
  int poolSize;
  int poolStride;
  int dimension;
  int S;
  MaxPoolingTriangularLayer(cudaMemStream &memStream, int poolSize,
                            int poolStride, int dimension);
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
