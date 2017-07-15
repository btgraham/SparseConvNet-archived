#pragma once
#include "SpatiallySparseLayer.h"

class ReallyConvolutionalLayer : public SpatiallySparseLayer {
private:
  int fs;
  RNG rng;
  float leaky;

public:
  int inSpatialSize;
  int outSpatialSize;
  int filterSize;
  int filterStride;
  int dimension;
  ActivationFunction fn;
  int nFeaturesIn;
  int nFeaturesOut;
  float dropout;
  int minActiveInputs;
  vectorCUDA<float> W;  // Weights
  vectorCUDA<float> MW; // momentum
  vectorCUDA<float> w;  // shrunk versions
  vectorCUDA<float> dw; // For backprop
  vectorCUDA<float> B;  // Weights
  vectorCUDA<float> MB; // momentum
  vectorCUDA<float> b;  // shrunk versions
  vectorCUDA<float> db; // For backprop
  ReallyConvolutionalLayer(cudaMemStream &memStream, int nFeaturesIn,
                           int nFeaturesOut, int filterSize, int filterStride,
                           int dimension, ActivationFunction fn, float dropout,
                           int minActiveInputs = 1, float poolingToFollow = 1);
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
  void scaleWeights(SpatiallySparseBatchInterface &input,
                    SpatiallySparseBatchInterface &output,
                    float &scalingUnderneath, bool topLayer);
  int calculateInputSpatialSize(int outputSpatialSize);
};
