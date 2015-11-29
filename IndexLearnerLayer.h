// See Ben Graham
// http://www2.warwick.ac.uk/fac/sci/statistics/staff/academic-research/graham/indexlearning.pdf
// and
// http://papers.nips.cc/paper/5548-discriminative-unsupervised-feature-learning-with-convolutional-neural-networks.pdf

#pragma once
#include <fstream>
#include "SpatiallySparseLayer.h"
#include "Rng.h"
#include "vectorCUDA.h"
#include <fstream>
#include <vector>

class IndexLearnerLayer : public SpatiallySparseLayer {
private:
  RNG rng;
  vectorCUDA<float> W; // Weights
  vectorCUDA<float> MW; // momentum
  vectorCUDA<float> w; // shrunk versions
  vectorCUDA<float> dw; // For backprop
public:
  std::vector<int> indexLearnerIndices; // Variable to deliver indices in use
                                        // for "double minibatch gradient
                                        // descent"
  int nFeaturesIn;
  int nFeaturesOut;
  float dropout;
  IndexLearnerLayer(cudaMemStream &memStream, int nFeaturesIn,
                    int nFeaturesOut);
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
  void loadWeightsFromStream(std::ifstream &f);
  void putWeightsToStream(std::ofstream &f);
  int calculateInputSpatialSize(int outputSpatialSize);
};

void IndexLearner(SpatiallySparseBatchInterface &input,
                  SpatiallySparseBatch &batch, int nTop,
                  cudaMemStream &memStream);
