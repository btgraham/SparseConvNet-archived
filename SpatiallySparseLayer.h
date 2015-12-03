#pragma once
#include <iostream>
#include "SpatiallySparseBatchInterface.h"
#include "SpatiallySparseBatch.h"

class SpatiallySparseLayer {
public:
  cudaMemStream &memStream;
  virtual void preprocess(SpatiallySparseBatch &batch,
                          SpatiallySparseBatchInterface &input,
                          SpatiallySparseBatchInterface &output) = 0;
  virtual void forwards(SpatiallySparseBatch &batch,
                        SpatiallySparseBatchInterface &input,
                        SpatiallySparseBatchInterface &output) = 0;
  virtual void scaleWeights(SpatiallySparseBatchInterface &input,
                            SpatiallySparseBatchInterface &output,
                            float &scalingUnderneath, bool topLayer);
  virtual void backwards(SpatiallySparseBatch &batch,
                         SpatiallySparseBatchInterface &input,
                         SpatiallySparseBatchInterface &output,
                         float learningRate, float momentum) = 0;
  SpatiallySparseLayer(cudaMemStream &memStream);
  ~SpatiallySparseLayer();
  virtual void loadWeightsFromStream(std::ifstream &f, bool momentum);
  virtual void putWeightsToStream(std::ofstream &f, bool momentum);
  virtual int calculateInputSpatialSize(int outputSpatialSize) = 0;
};
