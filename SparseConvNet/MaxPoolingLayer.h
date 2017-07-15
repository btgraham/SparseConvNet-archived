#pragma once
#include <iostream>
#include "SpatiallySparseLayer.h"
#include "Rng.h"

void maxPool(float *g1, float *g2, int *rules, int count, int sd, int nOut,
             int *d_choice, cudaMemStream &memStream);
void maxPoolBackProp(float *d1, float *d2, int count, int nOut, int *d_choice,
                     cudaMemStream &memStream);

class MaxPoolingLayer : public SpatiallySparseLayer {
public:
  int inSpatialSize;
  int outSpatialSize;
  int poolSize;
  int poolStride;
  int dimension;
  int sd;
  MaxPoolingLayer(cudaMemStream &memStream, int poolSize, int poolStride,
                  int dimension);
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

class PseudorandomOverlappingFractionalMaxPoolingLayer
    : public SpatiallySparseLayer {
  int sd;

public:
  int inSpatialSize;
  int outSpatialSize;
  float fmpShrink;
  int poolSize;
  int dimension;
  RNG rng;
  PseudorandomOverlappingFractionalMaxPoolingLayer(cudaMemStream &memStream,
                                                   int poolSize,
                                                   float fmpShrink,
                                                   int dimension);
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

class PseudorandomNonOverlappingFractionalMaxPoolingLayer
    : public SpatiallySparseLayer {
  int sd;

public:
  int inSpatialSize;
  int outSpatialSize;
  float fmpShrink;
  int poolSize;
  int dimension;
  RNG rng;
  PseudorandomNonOverlappingFractionalMaxPoolingLayer(cudaMemStream &memStream,
                                                      int poolSize,
                                                      float fmpShrink,
                                                      int dimension);
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

class RandomOverlappingFractionalMaxPoolingLayer : public SpatiallySparseLayer {
  int sd;

public:
  int inSpatialSize;
  int outSpatialSize;
  float fmpShrink;
  int poolSize;
  int dimension;
  RNG rng;
  RandomOverlappingFractionalMaxPoolingLayer(cudaMemStream &memStream,
                                             int poolSize, float fmpShrink,
                                             int dimension);
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

class RandomNonOverlappingFractionalMaxPoolingLayer
    : public SpatiallySparseLayer {
  int sd;

public:
  int inSpatialSize;
  int outSpatialSize;
  float fmpShrink;
  int poolSize;
  int dimension;
  RNG rng;
  RandomNonOverlappingFractionalMaxPoolingLayer(cudaMemStream &memStream,
                                                int poolSize, float fmpShrink,
                                                int dimension);
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
