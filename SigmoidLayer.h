#pragma once
#include "SpatiallySparseLayer.h"

void applySigmoid(SpatiallySparseBatchInterface &input,
                  SpatiallySparseBatchInterface &output, ActivationFunction fn,
                  cudaMemStream &memStream);
void applySigmoidBackProp(SpatiallySparseBatchInterface &input,
                          SpatiallySparseBatchInterface &output,
                          ActivationFunction fn, cudaMemStream &memStream);

class SigmoidLayer : public SpatiallySparseLayer {
public:
  ActivationFunction fn;
  SigmoidLayer(cudaMemStream &memStream, ActivationFunction fn);
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
