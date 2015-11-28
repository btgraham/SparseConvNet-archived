#pragma once
#include <iostream>
#include "SpatiallySparseLayer.h"
#include "Rng.h"

class TerminalPoolingLayer : public SpatiallySparseLayer {
public:
  int inSpatialSize;  //==poolSize.
  int outSpatialSize; //1
  int poolSize;
  int S;              //Maximum number of active sites, smaller than TERMINAL_POOLING_MAX_ACTIVE_SITES==1024
  TerminalPoolingLayer(cudaMemStream& memStream, int poolSize, int S);
  void preprocess
  (SpatiallySparseBatch &batch,
   SpatiallySparseBatchInterface &input,
   SpatiallySparseBatchInterface &output);
  void forwards(SpatiallySparseBatch &batch,
                SpatiallySparseBatchInterface &input,
                SpatiallySparseBatchInterface &output);
  void backwards(SpatiallySparseBatch &batch,
                 SpatiallySparseBatchInterface &input,
                 SpatiallySparseBatchInterface &output,
                 float learningRate,
                 float momentum);
  int calculateInputSpatialSize(int outputSpatialSize);
};
