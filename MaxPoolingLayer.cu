#include <iostream>
#include <cassert>
#include "utilities.h"
#include "cudaUtilities.h"
#include "MaxPoolingLayer.h"
#include "Regions.h"

template<int ps> __global__ void dMaxPool(float* g1, float* g2, int* rules, int nOut, int* d_choice) {
  __shared__ float s[ps][KERNELBLOCKSIZE];
  __shared__ float t[KERNELBLOCKSIZE];
  __shared__ int c[KERNELBLOCKSIZE];
  __shared__ int r[ps];
  int i=blockIdx.x*nOut;//for output
  for (int p=threadIdx.x;p<ps;p+=KERNELBLOCKSIZE)
    r[p]=rules[blockIdx.x*ps+p]*nOut;  //for input
  __syncthreads();
  for (int j=threadIdx.x;j<nOut;j+=KERNELBLOCKSIZE) {
    for (int p=0;p<ps;p++) {
      s[p][threadIdx.x]=(r[p]>=0)?g1[r[p]+j]:-10000000;
      __syncthreads();
    }
    for (int p=0;p<ps;p++)
      if (p==0 or t[threadIdx.x]<s[p][threadIdx.x]) {
        c[threadIdx.x]=r[p]+j;
        t[threadIdx.x]=s[p][threadIdx.x];
      }
    __syncthreads();
    g2[i+j]=t[threadIdx.x];
    d_choice[i+j]=c[threadIdx.x];
    __syncthreads();
  }
}

void maxPool(float* g1, float* g2, int* rules, int count, int sd, int nOut, int* d_choice) {
  //std::cout << g1 << " " << g2 << " " << rules << " " <<count << " " << sd << " " << nOut << " " << d_choice << std::endl;
  int processed=0;
  while (processed<count) {
    int batch=min(32768,count-processed);
    switch(sd) {
      //List of possible pooling regions:
      // powers of 2 for square grids,
      // powers of 3 for cubic grids, also
      // powers of 4, triangular numbers, tetrahedral numbers, etc
    case  3: dMaxPool< 3><<<batch,KERNELBLOCKSIZE,0,cnnMemStream->stream>>> (g1, g2+processed*nOut, rules+processed*sd, nOut, d_choice+processed*nOut); break;
    case  4: dMaxPool< 4><<<batch,KERNELBLOCKSIZE,0,cnnMemStream->stream>>> (g1, g2+processed*nOut, rules+processed*sd, nOut, d_choice+processed*nOut); break;
    case  5: dMaxPool< 5><<<batch,KERNELBLOCKSIZE,0,cnnMemStream->stream>>> (g1, g2+processed*nOut, rules+processed*sd, nOut, d_choice+processed*nOut); break;
    case  6: dMaxPool< 6><<<batch,KERNELBLOCKSIZE,0,cnnMemStream->stream>>> (g1, g2+processed*nOut, rules+processed*sd, nOut, d_choice+processed*nOut); break;
    case  8: dMaxPool< 8><<<batch,KERNELBLOCKSIZE,0,cnnMemStream->stream>>> (g1, g2+processed*nOut, rules+processed*sd, nOut, d_choice+processed*nOut); break;
    case  9: dMaxPool< 9><<<batch,KERNELBLOCKSIZE,0,cnnMemStream->stream>>> (g1, g2+processed*nOut, rules+processed*sd, nOut, d_choice+processed*nOut); break;
    case 10: dMaxPool<10><<<batch,KERNELBLOCKSIZE,0,cnnMemStream->stream>>> (g1, g2+processed*nOut, rules+processed*sd, nOut, d_choice+processed*nOut); break;
    case 15: dMaxPool<15><<<batch,KERNELBLOCKSIZE,0,cnnMemStream->stream>>> (g1, g2+processed*nOut, rules+processed*sd, nOut, d_choice+processed*nOut); break;
    case 16: dMaxPool<16><<<batch,KERNELBLOCKSIZE,0,cnnMemStream->stream>>> (g1, g2+processed*nOut, rules+processed*sd, nOut, d_choice+processed*nOut); break;
    case 20: dMaxPool<20><<<batch,KERNELBLOCKSIZE,0,cnnMemStream->stream>>> (g1, g2+processed*nOut, rules+processed*sd, nOut, d_choice+processed*nOut); break;
    case 25: dMaxPool<25><<<batch,KERNELBLOCKSIZE,0,cnnMemStream->stream>>> (g1, g2+processed*nOut, rules+processed*sd, nOut, d_choice+processed*nOut); break;
    case 27: dMaxPool<27><<<batch,KERNELBLOCKSIZE,0,cnnMemStream->stream>>> (g1, g2+processed*nOut, rules+processed*sd, nOut, d_choice+processed*nOut); break;
    case 35: dMaxPool<35><<<batch,KERNELBLOCKSIZE,0,cnnMemStream->stream>>> (g1, g2+processed*nOut, rules+processed*sd, nOut, d_choice+processed*nOut); break;
    case 36: dMaxPool<36><<<batch,KERNELBLOCKSIZE,0,cnnMemStream->stream>>> (g1, g2+processed*nOut, rules+processed*sd, nOut, d_choice+processed*nOut); break;
    case 49: dMaxPool<49><<<batch,KERNELBLOCKSIZE,0,cnnMemStream->stream>>> (g1, g2+processed*nOut, rules+processed*sd, nOut, d_choice+processed*nOut); break;
    default: std::cout << "Do some copying and pasting in " << __FILE__ << " line " << __LINE__ << " sd=" << sd << std::endl; exit(1); break;
    }
    processed+=batch;
  }
  cudaCheckError();
}

__global__ void dMaxPoolBackProp
(float* d1, float* d2, int nOut,int* d_choice) {
  //initialize d1 to zero first!!!
  int i=blockIdx.x*nOut;
  for (int j=i+threadIdx.x;j<i+nOut;j+=KERNELBLOCKSIZE) {
    atomicAdd(&d1[d_choice[j]],d2[j]);    //d1[d_choice[j]]=d2[j];
  }
}

void maxPoolBackProp(float* d1, float* d2, int count, int nOut, int* d_choice) {
  int processed=0;
  while (processed<count) {
    int batch=min(32768,count-processed);
    dMaxPoolBackProp<<<batch,KERNELBLOCKSIZE,0,cnnMemStream->stream>>> (d1, d2+processed*nOut, nOut, d_choice+processed*nOut);
    processed+=batch;
  }
  cudaCheckError();
}

//TODO: Refactor the different pooling classes somehow


MaxPoolingLayer::MaxPoolingLayer(int poolSize, int poolStride, int dimension)
  : poolSize(poolSize), poolStride(poolStride), dimension(dimension) {
  sd=ipow(poolSize,dimension);
  std::cout << "MaxPooling " << poolSize << " " << poolStride << std::endl;
}
void MaxPoolingLayer::preprocess
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
  RegularPoolingRegions regions(inSpatialSize, outSpatialSize,dimension,poolSize, poolStride);
  for (int item=0;item<batch.batchSize;item++)
    gridRules
      (input.grids[item],
       output.grids[item],
       regions,
       output.nSpatialSites,
       output.rules.hVector());
}
void MaxPoolingLayer::forwards
(SpatiallySparseBatch &batch,
 SpatiallySparseBatchInterface &input,
 SpatiallySparseBatchInterface &output) {
  output.sub->poolingChoices.resize(output.nSpatialSites*output.featuresPresent.size());
  output.sub->features.resize(output.nSpatialSites*output.featuresPresent.size());
  cudaCheckError();
  maxPool(input.sub->features.dPtr(),output.sub->features.dPtr(),output.rules.dPtr(),output.nSpatialSites,sd,output.featuresPresent.size(),output.sub->poolingChoices.dPtr());
  cudaCheckError();
}
void MaxPoolingLayer::backwards
(SpatiallySparseBatch &batch,
 SpatiallySparseBatchInterface &input,
 SpatiallySparseBatchInterface &output,
 float learningRate,
 float momentum) {
  if (input.backpropErrors) {
    input.sub->dfeatures.resize(input.nSpatialSites*input.featuresPresent.size());
    input.sub->dfeatures.setZero(*cnnMemStream);
    maxPoolBackProp
      (input.sub->dfeatures.dPtr(), output.sub->dfeatures.dPtr(), output.nSpatialSites, output.featuresPresent.size(), output.sub->poolingChoices.dPtr());
    // output.sub->features.resize(0);
    // output.sub->dfeatures.resize(0);
    // cudaCheckError();
  }
}
int MaxPoolingLayer::calculateInputSpatialSize(int outputSpatialSize) {
  outSpatialSize=outputSpatialSize;
  inSpatialSize=poolSize+(outputSpatialSize-1)*poolStride;
  std::cout << "(" << outSpatialSize <<"," <<inSpatialSize << ") ";
  return inSpatialSize;
}

PseudorandomOverlappingFractionalMaxPoolingLayer::PseudorandomOverlappingFractionalMaxPoolingLayer(int poolSize, float fmpShrink, int dimension) : poolSize(poolSize), fmpShrink(fmpShrink), dimension(dimension) {
  sd=ipow(poolSize,dimension);
  std::cout << "Pseudorandom overlapping Fractional Max Pooling " << fmpShrink << " " << poolSize << std::endl;
}
void PseudorandomOverlappingFractionalMaxPoolingLayer::preprocess
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
  PseudorandomOverlappingFractionalPoolingRegions regions(inSpatialSize, outSpatialSize,dimension, poolSize,rng);
  for (int item=0;item<batch.batchSize;item++)
    gridRules
      (input.grids[item],
       output.grids[item],
       regions,
       output.nSpatialSites,
       output.rules.hVector());
}
void PseudorandomOverlappingFractionalMaxPoolingLayer::forwards
(SpatiallySparseBatch &batch,
 SpatiallySparseBatchInterface &input,
 SpatiallySparseBatchInterface &output) {
  output.sub->poolingChoices.resize(output.nSpatialSites*output.featuresPresent.size());
  output.sub->features.resize(output.nSpatialSites*output.featuresPresent.size());
  cudaCheckError();
  maxPool(input.sub->features.dPtr(),output.sub->features.dPtr(),output.rules.dPtr(),output.nSpatialSites,sd,output.featuresPresent.size(),output.sub->poolingChoices.dPtr());
  cudaCheckError();
}
void PseudorandomOverlappingFractionalMaxPoolingLayer::backwards
(SpatiallySparseBatch &batch,
 SpatiallySparseBatchInterface &input,
 SpatiallySparseBatchInterface &output,
 float learningRate,
 float momentum) {
  if (input.backpropErrors) {
    input.sub->dfeatures.resize(input.nSpatialSites*input.featuresPresent.size());
    input.sub->dfeatures.setZero(*cnnMemStream);
    maxPoolBackProp
      (input.sub->dfeatures.dPtr(), output.sub->dfeatures.dPtr(), output.nSpatialSites, output.featuresPresent.size(), output.sub->poolingChoices.dPtr());
    // output.sub->features.resize(0);
    // output.sub->dfeatures.resize(0);
    // cudaCheckError();
  }
}
int PseudorandomOverlappingFractionalMaxPoolingLayer::calculateInputSpatialSize
(int outputSpatialSize) {
  outSpatialSize=outputSpatialSize;
  inSpatialSize=outputSpatialSize*fmpShrink+0.5;
  if (inSpatialSize==outputSpatialSize)
    inSpatialSize++;
  std::cout << "(" << outSpatialSize <<"," <<inSpatialSize << ") ";
  return inSpatialSize;
}

RandomOverlappingFractionalMaxPoolingLayer::RandomOverlappingFractionalMaxPoolingLayer
(int poolSize, float fmpShrink, int dimension)
  : poolSize(poolSize), fmpShrink(fmpShrink), dimension(dimension) {
  sd=ipow(poolSize,dimension);
  std::cout << "Random overlapping Fractional Max Pooling " << fmpShrink << " " << poolSize << std::endl;
}
void RandomOverlappingFractionalMaxPoolingLayer::preprocess
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
  RandomOverlappingFractionalPoolingRegions regions(inSpatialSize, outSpatialSize,dimension, poolSize,rng);
  for (int item=0;item<batch.batchSize;item++)
    gridRules
      (input.grids[item],
       output.grids[item],
       regions,
       output.nSpatialSites,
       output.rules.hVector());
}
void RandomOverlappingFractionalMaxPoolingLayer::forwards
(SpatiallySparseBatch &batch,
 SpatiallySparseBatchInterface &input,
 SpatiallySparseBatchInterface &output) {
  output.sub->poolingChoices.resize(output.nSpatialSites*output.featuresPresent.size());
  output.sub->features.resize(output.nSpatialSites*output.featuresPresent.size());
  cudaCheckError();
  maxPool(input.sub->features.dPtr(),output.sub->features.dPtr(),output.rules.dPtr(),output.nSpatialSites,sd,output.featuresPresent.size(),output.sub->poolingChoices.dPtr());
  cudaCheckError();
}
void RandomOverlappingFractionalMaxPoolingLayer::backwards
(SpatiallySparseBatch &batch,
 SpatiallySparseBatchInterface &input,
 SpatiallySparseBatchInterface &output,
 float learningRate,
 float momentum) {
  if (input.backpropErrors) {
    input.sub->dfeatures.resize(input.nSpatialSites*input.featuresPresent.size());
    input.sub->dfeatures.setZero(*cnnMemStream);
    maxPoolBackProp
      (input.sub->dfeatures.dPtr(), output.sub->dfeatures.dPtr(), output.nSpatialSites, output.featuresPresent.size(), output.sub->poolingChoices.dPtr());
    // output.sub->features.resize(0);
    // output.sub->dfeatures.resize(0);
    // cudaCheckError();
  }
}
int RandomOverlappingFractionalMaxPoolingLayer::calculateInputSpatialSize
(int outputSpatialSize) {
  outSpatialSize=outputSpatialSize;
  inSpatialSize=outputSpatialSize*fmpShrink+0.5;
  if (inSpatialSize==outputSpatialSize)
    inSpatialSize++;
  std::cout << "(" << outSpatialSize <<"," <<inSpatialSize << ") ";
  return inSpatialSize;
}
