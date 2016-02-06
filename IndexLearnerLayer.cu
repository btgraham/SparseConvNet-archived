#include "IndexLearnerLayer.h"
#include "SigmoidLayer.h"
#include <iostream>
#include <cassert>
#include "SoftmaxClassifier.h"
#include "NetworkInNetworkLayer.h"
#include "utilities.h"

__global__ void dGradientDescentShrunkMatrixNoMomentum(
    float *d_delta, float *d_weights, int nOut, int nOutDropout,
    int *inFeaturesPresent, int *outFeaturesPresent, float learningRate) {
  int i = blockIdx.x * nOutDropout;
  int ii = inFeaturesPresent[blockIdx.x] * nOut;
  for (int j = threadIdx.x; j < nOutDropout; j += KERNELBLOCKSIZE) {
    int jj = outFeaturesPresent[j];
    // no momentum, weight updated infrequently if the dataset is much larger
    // than each minibatch
    d_weights[ii + jj] -= learningRate * d_delta[i + j];
  }
}

IndexLearnerLayer::IndexLearnerLayer(cudaMemStream &memStream,
                                     cublasHandle_t &cublasHandle,
                                     int nFeaturesIn, int nFeaturesOut)
    : SpatiallySparseLayer(memStream), cublasHandle(cublasHandle),
      nFeaturesIn(nFeaturesIn), nFeaturesOut(nFeaturesOut) {
  std::cout << "IndexLearnerLayer" << std::endl;
  float scale = pow(6.0f / (nFeaturesIn + nFeaturesOut), 0.5f);
  W.resize(nFeaturesIn * nFeaturesOut);
  W.setZero(); // Uniform(-scale,scale);
  MW.resize(nFeaturesIn * nFeaturesOut);
  MW.setZero();
}
void IndexLearnerLayer::preprocess(SpatiallySparseBatch &batch,
                                   SpatiallySparseBatchInterface &input,
                                   SpatiallySparseBatchInterface &output) {
  if (batch.type == TRAINBATCH) {
    assert(input.nFeatures == nFeaturesIn);
    output.nFeatures = nFeaturesOut;
    output.spatialSize = input.spatialSize;
    output.nSpatialSites = input.nSpatialSites;
    output.grids = input.grids;
    output.backpropErrors = true;
  }
}

void IndexLearnerLayer::forwards(SpatiallySparseBatch &batch,
                                 SpatiallySparseBatchInterface &input,
                                 SpatiallySparseBatchInterface &output) {
  output.featuresPresent.hVector() = indexLearnerIndices;
  output.sub->features.resize(output.nSpatialSites *
                              output.featuresPresent.size());
  output.sub->dfeatures.resize(output.nSpatialSites *
                               output.featuresPresent.size());
  w.resize(input.featuresPresent.size() * output.featuresPresent.size());
  dShrinkMatrixForDropout << <input.featuresPresent.size(), KERNELBLOCKSIZE, 0,
                              memStream.stream>>>
      (W.dPtr(), w.dPtr(), input.featuresPresent.dPtr(),
       output.featuresPresent.dPtr(), output.nFeatures,
       output.featuresPresent.size());
  cudaCheckError();
  d_rowMajorSGEMM_alphaAB_betaC(
      cublasHandle, input.sub->features.dPtr(), w.dPtr(),
      output.sub->features.dPtr(), output.nSpatialSites,
      input.featuresPresent.size(), output.featuresPresent.size(), 1.0f, 0.0f,
      __FILE__, __LINE__);
  applySigmoid(output, output, SOFTMAX, memStream);
  cudaCheckError();
}
void IndexLearnerLayer::backwards(SpatiallySparseBatch &batch,
                                  SpatiallySparseBatchInterface &input,
                                  SpatiallySparseBatchInterface &output,
                                  float learningRate, float momentum) {
  applySigmoidBackProp(output, output, SOFTMAX, memStream);
  input.sub->dfeatures.resize(input.nSpatialSites *
                              input.featuresPresent.size());
  dw.resize(input.featuresPresent.size() * output.featuresPresent.size());
  d_rowMajorSGEMM_alphaAtB_betaC(
      cublasHandle, input.sub->features.dPtr(), output.sub->dfeatures.dPtr(),
      dw.dPtr(), input.featuresPresent.size(), output.nSpatialSites,
      output.featuresPresent.size(), 1.0, 0.0);
  cudaCheckError();

  if (input.backpropErrors) {
    d_rowMajorSGEMM_alphaABt_betaC(
        cublasHandle, output.sub->dfeatures.dPtr(), w.dPtr(),
        input.sub->dfeatures.dPtr(), output.nSpatialSites,
        output.featuresPresent.size(), input.featuresPresent.size(), 1.0, 0.0);
    cudaCheckError();
  }
  dGradientDescentShrunkMatrixNoMomentum
          << <input.featuresPresent.size(), KERNELBLOCKSIZE, 0,
              memStream.stream>>>
      (dw.dPtr(), W.dPtr(), output.nFeatures, output.featuresPresent.size(),
       input.featuresPresent.dPtr(), output.featuresPresent.dPtr(),
       learningRate);
  cudaCheckError();
}
void IndexLearnerLayer::loadWeightsFromStream(std::ifstream &f, bool momentum) {
  f.read((char *)&W.hVector()[0], sizeof(float) * W.size());
};
void IndexLearnerLayer::putWeightsToStream(std::ofstream &f, bool momentum) {
  f.write((char *)&W.hVector()[0], sizeof(float) * W.size());
};
int IndexLearnerLayer::calculateInputSpatialSize(int outputSpatialSize) {
  return outputSpatialSize;
}

void IndexLearner(SpatiallySparseBatchInterface &input,
                  SpatiallySparseBatch &batch, int nTop,
                  cudaMemStream &memStream) {
  assert(batch.batchSize == input.nSpatialSites);
  assert(ipow(batch.batchSize, 2) == input.sub->features.size());
  assert(batch.type == TRAINBATCH);

  float *probs = &input.sub->features.hVector()[0];
  for (int i = 0; i < batch.batchSize; ++i)
    batch.probabilities.push_back(std::vector<float>(
        probs + i * batch.batchSize, probs + (i + 1) * batch.batchSize));
  for (int i = 0; i < batch.batchSize; i++)
    batch.predictions.push_back(vectorTopIndices(batch.probabilities[i], nTop));

  batch.mistakes += batch.batchSize;
  for (int i = 0; i < batch.batchSize; i++) {
    batch.negativeLogLikelihood -= log(max(batch.probabilities[i][i], 1.0e-15));
    for (int j = 0; j < nTop; j++) {
      if (batch.predictions[i][j] == i) {
        batch.mistakes--;
      }
    }
  }
  // Begin backprop. Top layer: d Cost / d SoftmaxInput
  vectorCUDA<int> labels;
  labels.hVector() = range(batch.batchSize);
  input.sub->dfeatures.resize(input.nSpatialSites *
                              input.featuresPresent.size());
  dDerivativeOfCostWRTpreSoftmaxTopLevelWeights
          << <1, NTHREADS, 0, memStream.stream>>>
      (batch.batchSize, input.sub->dfeatures.dPtr(), input.sub->features.dPtr(),
       labels.dPtr(), batch.batchSize);
  cudaCheckError();
}
