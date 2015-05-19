#pragma once
#include <fstream>
#include "SpatiallySparseLayer.h"
#include "Rng.h"

#ifndef NAG_MU
#define NAG_MU 0.99
#endif

class NetworkInNetworkLayer : public SpatiallySparseLayer {
private:
  RNG rng;
public:
  vectorCUDA<float> W; //Weights
  vectorCUDA<float> MW; //momentum
  vectorCUDA<float> w; //shrunk versions
  vectorCUDA<float> dw; //For backprop
  vectorCUDA<float> B; //Weights
  vectorCUDA<float> MB; //momentum
  vectorCUDA<float> b; //shrunk versions
  vectorCUDA<float> db; //For backprop
  ActivationFunction fn;
  int nFeaturesIn;
  int nFeaturesOut;
  float dropout;
  NetworkInNetworkLayer(int nFeaturesIn, int nFeaturesOut,
                        float dropout=0,ActivationFunction fn=NOSIGMOID,
                        float alpha=1//used to determine intialization weights only
                        );
  void preprocess
  (SpatiallySparseBatch &batch,
   SpatiallySparseBatchInterface &input,
   SpatiallySparseBatchInterface &output);
  void forwards
  (SpatiallySparseBatch &batch,
   SpatiallySparseBatchInterface &input,
   SpatiallySparseBatchInterface &output);
  void scaleWeights
  (SpatiallySparseBatchInterface &input,
   SpatiallySparseBatchInterface &output);
  void backwards
  (SpatiallySparseBatch &batch,
   SpatiallySparseBatchInterface &input,
   SpatiallySparseBatchInterface &output,
   float learningRate);
  void loadWeightsFromStream(std::ifstream &f);
  void putWeightsToStream(std::ofstream &f);
  int calculateInputSpatialSize(int outputSpatialSize);
};

__global__ void dShrinkMatrixForDropout
(float* m, float* md,
 int* inFeaturesPresent, int* outFeaturesPresent,
 int nOut, int nOutDropout);
__global__ void dGradientDescentShrunkMatrix
(float* d_delta, float* d_momentum, float* d_weights,
 int nOut, int nOutDropout,
 int* inFeaturesPresent, int* outFeaturesPresent,
 float learningRate);
