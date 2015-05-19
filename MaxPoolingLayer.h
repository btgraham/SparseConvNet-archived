#pragma once
#include <iostream>
#include "SpatiallySparseLayer.h"
#include "Rng.h"

void maxPool(float* g1, float* g2, int* rules, int count, int sd, int nOut, int* d_choice);
void maxPoolBackProp(float* d1, float* d2, int count, int nOut, int* d_choice);

//TODO: Refactor the different pooling classes somehow


class MaxPoolingLayer : public SpatiallySparseLayer {
public:
  int inSpatialSize;
  int outSpatialSize;
  int poolSize;
  int poolStride;
  int dimension;
  int sd;
  MaxPoolingLayer(int poolSize, int poolStride, int dimension);
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
                 float learningRate);
  int calculateInputSpatialSize(int outputSpatialSize);
};

class PseudorandomOverlappingFractionalMaxPoolingLayer : public SpatiallySparseLayer {
  int sd;
public:
  int inSpatialSize;
  int outSpatialSize;
  float fmpShrink;
  int poolSize;
  int dimension;
  RNG rng;
  PseudorandomOverlappingFractionalMaxPoolingLayer(int poolSize, float fmpShrink, int dimension);
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
                 float learningRate);
  int calculateInputSpatialSize(int outputSpatialSize);
};

class RandomOverlappingFractionalMaxPoolingLayer : public SpatiallySparseLayer {
  int sd;
public:
  int inSpatialSize;
  int outSpatialSize;
  float fmpShrink;
  int poolSize;
  int dimension;
  RNG rng;
  RandomOverlappingFractionalMaxPoolingLayer(int poolSize, float fmpShrink, int dimension);
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
   float learningRate);
  int calculateInputSpatialSize(int outputSpatialSize);
};
