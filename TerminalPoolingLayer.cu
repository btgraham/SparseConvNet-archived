//Average everything that makes it to the final layer

#define TERMINAL_POOLING_MAX_ACTIVE_SITES 1024
#include <iostream>
#include "utilities.h"
#include "cudaUtilities.h"
#include "TerminalPoolingLayer.h"

void terminalGridPoolingRules
(SparseGrid &inputGrid,
 SparseGrid &outputGrid,
 int S,
 int &nOutputSpatialSites,
 std::vector<int>& rules) {
  assert(inputGrid.mp.size()>0); //Danger, total loss of information
  assert(inputGrid.mp.size()<=TERMINAL_POOLING_MAX_ACTIVE_SITES); //Upper bound for ease of kernel memory management
  for (SparseGridIter iter = inputGrid.mp.begin(); iter != inputGrid.mp.end(); ++iter)
    rules.push_back(iter->second);
  outputGrid.mp[0]=nOutputSpatialSites++;
  rules.resize(S*nOutputSpatialSites,-1); //pad with -1 values
}

__global__ void dTerminalPool(float* g1, float* g2, int* rules, int nOut, int ps2) {
  __shared__ int r[TERMINAL_POOLING_MAX_ACTIVE_SITES];
  int i=blockIdx.x*nOut;//for output g2
  for (int p=threadIdx.x;p<ps2;p+=KERNELBLOCKSIZE)
    r[p]=rules[blockIdx.x*ps2+p]*nOut;  //for input g1
  __syncthreads();
  for (int j=threadIdx.x;j<nOut;j+=KERNELBLOCKSIZE) { //nOut is a multiple of KERNELBLOCKSIZE!!!
    float t=0;
    int p=0;
    for (;p<ps2 and r[p]>=0;p++) {
      t+=g1[r[p]+j];
    }
    g2[i+j]=t/p;
  }
}

void terminalPool(float* g1, float* g2, int* rules, int count, int ps2, int nOut) {
  int processed=0;
  assert(ps2<=TERMINAL_POOLING_MAX_ACTIVE_SITES);// if ps2>KERNELBLOCKSIZE, i.e. if poolSize>32, allocate more memory in dTerminalPool and dTerminalPoolBackProp
  while (processed<count) {
    int batch=min(32768,count-processed);
    dTerminalPool<<<batch,KERNELBLOCKSIZE,0,cnnMemStream->stream>>> (g1, g2+processed*nOut, rules+processed*ps2, nOut, ps2);
    processed+=batch;
  }
  cudaCheckError();
}


__global__ void dTerminalPoolBackProp(float* d1, float* d2, int* rules, int nOut, int ps2) {
  __shared__ int r[TERMINAL_POOLING_MAX_ACTIVE_SITES];  //Allocate at least size ps2 !!!!!!!!!!!
  int i=blockIdx.x*nOut;//for input d2
  for (int p=threadIdx.x;p<ps2;p+=KERNELBLOCKSIZE) {
    r[p]=rules[blockIdx.x*ps2+p]*nOut;  //for output d1
  }
  __syncthreads();
  int maxP=0;
  while (maxP<ps2 and r[maxP]>=0)
    ++maxP;
  __syncthreads();  //delete line??
  for (int j=threadIdx.x;j<nOut;j+=KERNELBLOCKSIZE) {
    float t=d2[i+j]/maxP;
    for (int p=0;p<maxP;p++) {
      d1[r[p]+j]=t;
    }
  }
}

void terminalPoolBackProp(float* d1, float* d2, int* rules, int count, int nOut, int ps2) {
  int processed=0;
  while (processed<count) {
    int batch=min(32768,count-processed);
    dTerminalPoolBackProp<<<batch,KERNELBLOCKSIZE,0,cnnMemStream->stream>>> (d1, d2+processed*nOut, rules+processed*ps2, nOut, ps2);
    processed+=batch;
  }
  cudaCheckError();
}

TerminalPoolingLayer::TerminalPoolingLayer(int poolSize, int S)
  : inSpatialSize(poolSize), outSpatialSize(1), poolSize(poolSize), S(S) {
  std::cout << "TerminalPooling " << poolSize << " " << S << std::endl;
}
void TerminalPoolingLayer::preprocess
(SpatiallySparseBatch &batch,
 SpatiallySparseBatchInterface &input,
 SpatiallySparseBatchInterface &output) {
  assert(input.spatialSize==inSpatialSize);
  output.nFeatures=input.nFeatures;
  output.featuresPresent.hVector()=input.featuresPresent.hVector();
  output.spatialSize=outSpatialSize;
  output.nSpatialSites=0;
  output.grids.resize(batch.batchSize);
  output.backpropErrors=input.backpropErrors;
  for (int item=0;item<batch.batchSize;item++)
    terminalGridPoolingRules
      (input.grids[item],
       output.grids[item],
       S,
       output.nSpatialSites,
       output.rules.hVector());
}
void TerminalPoolingLayer::forwards
(SpatiallySparseBatch &batch,
 SpatiallySparseBatchInterface &input,
 SpatiallySparseBatchInterface &output) {
  output.sub->poolingChoices.resize(output.nSpatialSites*output.featuresPresent.size());
  output.sub->features.resize(output.nSpatialSites*output.featuresPresent.size());
  cudaCheckError();
  terminalPool(input.sub->features.dPtr(),output.sub->features.dPtr(),output.rules.dPtr(),output.nSpatialSites,S,output.featuresPresent.size());
  cudaCheckError();
}
void TerminalPoolingLayer::backwards
(SpatiallySparseBatch &batch,
 SpatiallySparseBatchInterface &input,
 SpatiallySparseBatchInterface &output,
 float learningRate) {
  if (input.backpropErrors) {
    input.sub->dfeatures.resize(input.nSpatialSites*input.featuresPresent.size());
    input.sub->dfeatures.setZero();
    terminalPoolBackProp
      (input.sub->dfeatures.dPtr(), output.sub->dfeatures.dPtr(), output.rules.dPtr(),output.nSpatialSites, output.featuresPresent.size(),S);
    // output.sub->features.resize(0);
    // output.sub->dfeatures.resize(0);
    // cudaCheckError();
  }
}
int TerminalPoolingLayer::calculateInputSpatialSize(int outputSpatialSize) {
  assert(outputSpatialSize==1);
  std::cout << "(" << outSpatialSize <<"," <<inSpatialSize << ") ";
  return inSpatialSize;
}
