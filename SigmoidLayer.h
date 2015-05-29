#pragma once
#include "SpatiallySparseLayer.h"

void applySigmoid(SpatiallySparseBatchInterface& input, SpatiallySparseBatchInterface& output, ActivationFunction fn);
void applySigmoidBackProp(SpatiallySparseBatchInterface& input, SpatiallySparseBatchInterface& output, ActivationFunction fn);

class SigmoidLayer : public SpatiallySparseLayer {
public:
  ActivationFunction fn;
  SigmoidLayer(ActivationFunction fn);
  void preprocess
  (SpatiallySparseBatch &batch,
   SpatiallySparseBatchInterface &input,
   SpatiallySparseBatchInterface &output);
  void forwards
  (SpatiallySparseBatch &batch,
   SpatiallySparseBatchInterface &input,
   SpatiallySparseBatchInterface &output);
  void backwards
  (SpatiallySparseBatch &batch,
   SpatiallySparseBatchInterface &input,
   SpatiallySparseBatchInterface &output,
   float learningRate,
   float momentum);
  int calculateInputSpatialSize(int outputSpatialSize);
};
