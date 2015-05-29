#pragma once
#include <iostream>
#include "SpatiallySparseBatchInterface.h"
#include "SpatiallySparseBatch.h"

class SpatiallySparseLayer {
public:
  SpatiallySparseBatchSubInterface sub;
  virtual void preprocess
  (SpatiallySparseBatch &batch,
   SpatiallySparseBatchInterface &input,
   SpatiallySparseBatchInterface &output) = 0;
  virtual void forwards
  (SpatiallySparseBatch &batch,
   SpatiallySparseBatchInterface &input,
   SpatiallySparseBatchInterface &output) = 0;
  virtual void scaleWeights
  (SpatiallySparseBatchInterface &input, SpatiallySparseBatchInterface &output);
  virtual void backwards
  (SpatiallySparseBatch &batch,
   SpatiallySparseBatchInterface &input,
   SpatiallySparseBatchInterface &output,
   float learningRate,
   float momentum) = 0;
  virtual void loadWeightsFromStream(std::ifstream &f);
  virtual void putWeightsToStream(std::ofstream &f);
  virtual int calculateInputSpatialSize(int outputSpatialSize) = 0;
};
