#pragma once
#include <fstream>
#include "SpatiallySparseLayer.h"
#include "Rng.h"

class NetworkInNetworkLayer : public SpatiallySparseLayer {
private:
  RNG rng;
  cublasHandle_t &cublasHandle;

public:
  vectorCUDA<float> W;  // Weights
  vectorCUDA<float> MW; // momentum
  vectorCUDA<float> w;  // shrunk versions
  vectorCUDA<float> dw; // For backprop
  vectorCUDA<float> B;  // Weights
  vectorCUDA<float> MB; // momentum
  vectorCUDA<float> b;  // shrunk versions
  vectorCUDA<float> db; // For backprop
  ActivationFunction fn;
  int nFeaturesIn;
  int nFeaturesOut;
  float dropout;
  NetworkInNetworkLayer(
      cudaMemStream &memStream, cublasHandle_t &cublasHandle, int nFeaturesIn,
      int nFeaturesOut, float dropout = 0, ActivationFunction fn = NOSIGMOID,
      float alpha = 1 // used to determine intialization weights only
      );
  void preprocess(SpatiallySparseBatch &batch,
                  SpatiallySparseBatchInterface &input,
                  SpatiallySparseBatchInterface &output);
  void forwards(SpatiallySparseBatch &batch,
                SpatiallySparseBatchInterface &input,
                SpatiallySparseBatchInterface &output);
  void scaleWeights(SpatiallySparseBatchInterface &input,
                    SpatiallySparseBatchInterface &output,
                    float &scalingUnderneath, bool topLayer);
  void backwards(SpatiallySparseBatch &batch,
                 SpatiallySparseBatchInterface &input,
                 SpatiallySparseBatchInterface &output, float learningRate,
                 float momentum);
  void loadWeightsFromStream(std::ifstream &f, bool momentum);
  void putWeightsToStream(std::ofstream &f, bool momentum);
  int calculateInputSpatialSize(int outputSpatialSize);
};

__global__ void dShrinkMatrixForDropout(float *m, float *md,
                                        int *inFeaturesPresent,
                                        int *outFeaturesPresent, int nOut,
                                        int nOutDropout);
__global__ void dShrinkVectorForDropout(float *m, float *md,
                                        int *outFeaturesPresent, int nOut,
                                        int nOutDropout);
__global__ void dGradientDescent(float *d_delta, float *d_momentum,
                                 float *d_weights, int nOut, float learningRate,
                                 float momentum);
__global__ void
dGradientDescentShrunkMatrix(float *d_delta, float *d_momentum,
                             float *d_weights, int nOut, int nOutDropout,
                             int *inFeaturesPresent, int *outFeaturesPresent,
                             float learningRate, float momentum);
__global__ void dGradientDescentShrunkVector(float *d_delta, float *d_momentum,
                                             float *d_weights, int nOut,
                                             int nOutDropout,
                                             int *outFeaturesPresent,
                                             float learningRate,
                                             float momentum);
void replicateArray(float *src, float *dst, int nRows, int nColumns,
                    cudaMemStream &memStream);
void columnSum(float *matrix, float *target, int nRows, int nColumns,
               cudaMemStream &memStream);
