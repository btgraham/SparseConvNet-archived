//This does not really implement a convolution. It just gathers data together in prepartion for matrix muliplictation. "Proper convolution" = "ConvolutionalLayer" + "NetworkInNetworkLayer"

#include "ConvolutionalLayer.h"
#include <iostream>
#include <vector>
#include "cudaUtilities.h"
#include "utilities.h"
#include "Regions.h"

__global__ void dPropForwardToMatrixMultiplyInput
(float* d_features, float* d_convolved, int* rules, int count, int nIn) {
  __shared__ int r[KERNELBLOCKSIZE];
  for (int c=blockIdx.x*KERNELBLOCKSIZE; c<count; c+=(1<<12)*KERNELBLOCKSIZE) {
    int n=min(KERNELBLOCKSIZE,count-c);
    r[threadIdx.x]=(threadIdx.x<n)?rules[c+threadIdx.x]*nIn:0;
    __syncthreads();
    for (int q=0;q<n;q++) {
      int rq=r[q];
      int i=(c+q)*nIn;
      for (int j=threadIdx.x;j<nIn;j+=KERNELBLOCKSIZE) {
        d_convolved[i+j]=(rq>=0)?d_features[rq+j]:0;    //If padding is used, check rq!=-1
      }
    }
    __syncthreads();
  }
}
void propForwardToMatrixMultiply(float* inFeatures, float* outFeatures, int* rules, int count, int nIn) {
  assert(count>0);
  int batch=min(1<<12,(count+KERNELBLOCKSIZE-1)/KERNELBLOCKSIZE);
  dPropForwardToMatrixMultiplyInput<<<batch,KERNELBLOCKSIZE,0,cnnMemStream->stream>>>
    (inFeatures,
     outFeatures,
     rules,
     count, nIn);
  cudaCheckError();
}
__global__ void dPropBackwardFromMatrixMultiplyOutput
(float* d_deltaGrid, float* d_deltaConvolved, int* rules, int count, int nIn) {
  __shared__ int r[KERNELBLOCKSIZE];
  for (int c=blockIdx.x*KERNELBLOCKSIZE; c<count; c+=(1<<12)*KERNELBLOCKSIZE) {
    int n=min(KERNELBLOCKSIZE,count-c);
    r[threadIdx.x]=(threadIdx.x<n)?rules[c+threadIdx.x]*nIn:0;
    __syncthreads();
    for (int q=0;q<n;q++) {
      int rq=r[q];
      int i=(c+q)*nIn;
      for (int j=threadIdx.x;j<nIn;j+=KERNELBLOCKSIZE) {
        if (/*d_deltaConvolved[i+j]!=0 and*/ rq>=0)
          atomicAdd(&d_deltaGrid[rq+j],d_deltaConvolved[i+j]);
      }
    }
    __syncthreads();
  }
}
void propBackwardFromMatrixMultiply(float* inDFeatures, float* outDFeatures, int* rules, int count, int nIn) {
  assert(count>0);
  int batch=min(1<<12,(count+KERNELBLOCKSIZE-1)/KERNELBLOCKSIZE);
  dPropBackwardFromMatrixMultiplyOutput<<<batch,KERNELBLOCKSIZE,0,cnnMemStream->stream>>>
    (inDFeatures,
     outDFeatures,
     rules,
     count, nIn);
  cudaCheckError();
}

template <typename t> void convolutionFeaturesPresent(std::vector<t>& d_src, std::vector<t>& d_dest, int nf, int nfp, int nCopies) {
  for (int i=0;i<nfp*nCopies;++i) {
    d_dest[i]=d_src[i%nfp]+nf*(i/nfp);
  }
}
template void convolutionFeaturesPresent<int>(std::vector<int>& d_src, std::vector<int>& d_dest, int nf, int nfp, int nCopies);

ConvolutionalLayer::ConvolutionalLayer(int filterSize,
                                       int filterStride,
                                       int dimension,
                                       int nFeaturesIn,
                                       int minActiveInputs) :
  filterSize(filterSize),
  filterStride(filterStride),
  dimension(dimension),
  nFeaturesIn(nFeaturesIn),
  minActiveInputs(minActiveInputs) {
  fs=ipow(filterSize,dimension);
  nFeaturesOut=fs*nFeaturesIn;
  std::cout << "Convolution "
            << filterSize <<"^" <<dimension<< "x"<< nFeaturesIn
            << "->" << nFeaturesOut;
  if (filterStride>1)
    std::cout << " stride:" << filterStride;
  if (minActiveInputs>1)
    std::cout << " minActiveInputs:"  << minActiveInputs;
  std::cout << std::endl;
  }
void ConvolutionalLayer::preprocess
(SpatiallySparseBatch &batch,
 SpatiallySparseBatchInterface &input,
 SpatiallySparseBatchInterface &output) {
  output.nFeatures=nFeaturesOut;
  assert(input.nFeatures==nFeaturesIn);
  assert(input.spatialSize>=filterSize);
  assert((input.spatialSize-filterSize)%filterStride==0);
  output.spatialSize=(input.spatialSize-filterSize)/filterStride+1;
  output.nSpatialSites=0;
  output.grids.resize(batch.batchSize);
  output.backpropErrors=input.backpropErrors;
  RegularPoolingRegions regions(inSpatialSize, outSpatialSize,dimension,filterSize, filterStride);
  for (int item=0;item<batch.batchSize;item++) {
    gridRules(input.grids[item],
              output.grids[item],
              regions,
              output.nSpatialSites,
              output.rules.hVector(),
              minActiveInputs);
  }
  output.featuresPresent.copyToCPU();
  output.featuresPresent.resize(input.featuresPresent.size()*fs);
  convolutionFeaturesPresent(input.featuresPresent.hVector(), output.featuresPresent.hVector(), input.nFeatures, input.featuresPresent.size(), fs);
}
void ConvolutionalLayer::forwards
(SpatiallySparseBatch &batch,
 SpatiallySparseBatchInterface &input,
 SpatiallySparseBatchInterface &output) {
  output.sub->features.resize(output.nSpatialSites*output.featuresPresent.size());
  propForwardToMatrixMultiply(input.sub->features.dPtr(),
                              output.sub->features.dPtr(),
                              output.rules.dPtr(),
                              output.nSpatialSites*fs,
                              input.featuresPresent.size());
}
void ConvolutionalLayer::backwards
(SpatiallySparseBatch &batch,
 SpatiallySparseBatchInterface &input,
 SpatiallySparseBatchInterface &output,
 float learningRate,
 float momentum) {
  if (input.backpropErrors) {
    input.sub->dfeatures.resize(input.nSpatialSites*input.featuresPresent.size());
    input.sub->dfeatures.setZero(*cnnMemStream);
    propBackwardFromMatrixMultiply(input.sub->dfeatures.dPtr(),
                                   output.sub->dfeatures.dPtr(),
                                   output.rules.dPtr(),
                                   output.nSpatialSites*fs,
                                   input.featuresPresent.size());
    // output.sub->features.resize(0);
    // output.sub->dfeatures.resize(0);
    // cudaCheckError();
  }
}
int ConvolutionalLayer::calculateInputSpatialSize(int outputSpatialSize) {
  outSpatialSize=outputSpatialSize;
  inSpatialSize=filterSize+(outputSpatialSize-1)*filterStride;
  return inSpatialSize;
}
