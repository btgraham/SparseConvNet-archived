// Average everything that makes it to the final layer

#include <iostream>
#include <cassert>
#include "utilities.h"
#include "TerminalPoolingLayer.h"

void terminalGridPoolingRules(SparseGrid &inputGrid, SparseGrid &outputGrid,
                              int S, int &nOutputSpatialSites,
                              std::vector<int> &rules) {
  // std::cout << inputGrid.mp.size() << std::endl;
  if (inputGrid.mp.size() == 0) { // Danger, total loss of information
    rules.push_back(inputGrid.backgroundCol);
  } else {
    for (auto iter = inputGrid.mp.begin(); iter != inputGrid.mp.end(); ++iter)
      rules.push_back(iter->second);
  }
  outputGrid.mp[0] = nOutputSpatialSites++;
  rules.resize(S * nOutputSpatialSites, -1); // pad with -1 values
}

__global__ void dTerminalPool(float *g1, float *g2, int *rules, int nOut,
                              int S) {
  int i = blockIdx.x * nOut; // for output g2
  for (int j = threadIdx.x; j < nOut;
       j += KERNELBLOCKSIZE) { // nOut is a multiple of KERNELBLOCKSIZE!!!
    float t = 0;
    int p = 0;
    for (; p < S and rules[blockIdx.x * S + p] >= 0; p++) {
      t += g1[rules[blockIdx.x * S + p] * nOut + j];
    }
    g2[i + j] = t / p;
  }
}

void terminalPool(float *g1, float *g2, int *rules, int count, int S, int nOut,
                  cudaMemStream &memStream) {
  int processed = 0;
  while (processed < count) {
    int batch = min(32768, count - processed);
    dTerminalPool << <batch, KERNELBLOCKSIZE, 0, memStream.stream>>>
        (g1, g2 + processed * nOut, rules + processed * S, nOut, S);
    processed += batch;
  }
  cudaCheckError();
}

__global__ void dTerminalPoolBackProp(float *d1, float *d2, int *rules,
                                      int nOut, int S) {
  int i = blockIdx.x * nOut; // for input d2
  int maxP = 0;
  while (maxP < S and rules[blockIdx.x * S + maxP] >= 0)
    ++maxP;
  __syncthreads(); // delete line??
  for (int j = threadIdx.x; j < nOut; j += KERNELBLOCKSIZE) {
    float t = d2[i + j] / maxP;
    for (int p = 0; p < maxP; p++) {
      d1[rules[blockIdx.x * S + p] * nOut + j] = t;
    }
  }
}

void terminalPoolBackProp(float *d1, float *d2, int *rules, int count, int nOut,
                          int S, cudaMemStream &memStream) {
  int processed = 0;
  while (processed < count) {
    int batch = min(32768, count - processed);
    dTerminalPoolBackProp << <batch, KERNELBLOCKSIZE, 0, memStream.stream>>>
        (d1, d2 + processed * nOut, rules + processed * S, nOut, S);
    processed += batch;
  }
  cudaCheckError();
}

TerminalPoolingLayer::TerminalPoolingLayer(cudaMemStream &memStream,
                                           int poolSize, int S)
    : SpatiallySparseLayer(memStream), inSpatialSize(poolSize),
      outSpatialSize(1), poolSize(poolSize), S(S) {
  std::cout << "TerminalPooling " << poolSize << " " << S << std::endl;
}
void TerminalPoolingLayer::preprocess(SpatiallySparseBatch &batch,
                                      SpatiallySparseBatchInterface &input,
                                      SpatiallySparseBatchInterface &output) {
  assert(input.spatialSize == inSpatialSize);
  output.nFeatures = input.nFeatures;
  output.featuresPresent.hVector() = input.featuresPresent.hVector();
  output.spatialSize = outSpatialSize;
  output.nSpatialSites = 0;
  output.grids.resize(batch.batchSize);
  size_t s = 1;
  for (int i = 0; i < batch.batchSize; ++i)
    s = std::max(s, input.grids[i].mp.size());
  output.backpropErrors = input.backpropErrors;
  for (int item = 0; item < batch.batchSize; item++)
    terminalGridPoolingRules(input.grids[item], output.grids[item], s,
                             output.nSpatialSites, output.rules.hVector());
}
void TerminalPoolingLayer::forwards(SpatiallySparseBatch &batch,
                                    SpatiallySparseBatchInterface &input,
                                    SpatiallySparseBatchInterface &output) {
  output.sub->poolingChoices.resize(output.nSpatialSites *
                                    output.featuresPresent.size());
  output.sub->features.resize(output.nSpatialSites *
                              output.featuresPresent.size());
  cudaCheckError();
  terminalPool(input.sub->features.dPtr(), output.sub->features.dPtr(),
               output.rules.dPtr(), output.nSpatialSites,
               output.rules.size() / batch.batchSize,
               output.featuresPresent.size(), memStream);
  cudaCheckError();
}
void TerminalPoolingLayer::backwards(SpatiallySparseBatch &batch,
                                     SpatiallySparseBatchInterface &input,
                                     SpatiallySparseBatchInterface &output,
                                     float learningRate, float momentum) {
  if (input.backpropErrors) {
    input.sub->dfeatures.resize(input.nSpatialSites *
                                input.featuresPresent.size());
    input.sub->dfeatures.setZero();
    terminalPoolBackProp(input.sub->dfeatures.dPtr(),
                         output.sub->dfeatures.dPtr(), output.rules.dPtr(),
                         output.nSpatialSites, output.featuresPresent.size(),
                         output.rules.size() / batch.batchSize, memStream);
  }
}
int TerminalPoolingLayer::calculateInputSpatialSize(int outputSpatialSize) {
  assert(outputSpatialSize == 1);
  std::cout << "-(TP)-" << inSpatialSize;
  return inSpatialSize;
}
