#include "ConvolutionalTriangularLayer.h"
#include "ConvolutionalLayer.h"
#include <iostream>
#include "cudaUtilities.h"
#include "utilities.h"
#include "Regions.h"

ConvolutionalTriangularLayer::ConvolutionalTriangularLayer
(int filterSize, int filterStride, int dimension, int nFeaturesIn, int lPad, int rPad)
  : filterSize(filterSize), filterStride(filterStride), dimension(dimension),
    nFeaturesIn(nFeaturesIn), lPad(lPad), rPad(rPad) {
  fs=triangleSize(filterSize, dimension);
  nFeaturesOut=fs*nFeaturesIn;
  std::cout << dimension << "D ConvolutionalTriangularLayer side-length=" << filterSize
            << " " << nFeaturesIn << "x" << fs << "->" << nFeaturesOut;
  if (filterStride>1)
    std::cout << ", stride " << filterStride;
  std::cout << std::endl;
}
void ConvolutionalTriangularLayer::preprocess
(SpatiallySparseBatch &batch,
 SpatiallySparseBatchInterface &input,
 SpatiallySparseBatchInterface &output) {
  assert(input.nFeatures==nFeaturesIn);
  assert(input.spatialSize+lPad+rPad>=filterSize);
  assert((input.spatialSize+lPad+rPad-filterSize)%filterStride==0);
  output.nFeatures=nFeaturesOut;
  output.spatialSize=(input.spatialSize+lPad+rPad-filterSize)/filterStride+1;
  output.nSpatialSites=0;
  output.grids.resize(batch.batchSize);
  output.backpropErrors=input.backpropErrors;
  PaddedPoolingRegionsTriangular regions(inSpatialSize, outSpatialSize,dimension,filterSize, filterStride, lPad, rPad);
  for (int item=0;item<batch.batchSize;item++)
    gridRulesTriangular(input.grids[item],
                        output.grids[item],
                        regions,
                        output.nSpatialSites,
                        output.rules.hVector());
  output.featuresPresent.copyToCPU();
  output.featuresPresent.resize(input.featuresPresent.size()*fs);
  convolutionFeaturesPresent(input.featuresPresent.hVector(), output.featuresPresent.hVector(), input.nFeatures, input.featuresPresent.size(), fs);

}
void ConvolutionalTriangularLayer::forwards
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
void ConvolutionalTriangularLayer::backwards
(SpatiallySparseBatch &batch,
 SpatiallySparseBatchInterface &input,
 SpatiallySparseBatchInterface &output,
 float learningRate) {
  if (input.backpropErrors) {
    input.sub->dfeatures.resize(input.nSpatialSites*input.featuresPresent.size());
    input.sub->dfeatures.setZero(*cnnMemStream);
    propBackwardFromMatrixMultiply(input.sub->dfeatures.dPtr(),
                                   output.sub->dfeatures.dPtr(),
                                   output.rules.dPtr(),
                                   output.nSpatialSites*fs,
                                   input.featuresPresent.size());
  }
}
int ConvolutionalTriangularLayer::calculateInputSpatialSize(int outputSpatialSize) {
  outSpatialSize=outputSpatialSize;
  inSpatialSize=filterSize+(outputSpatialSize-1)*filterStride-lPad-rPad;
  std::cout << "(" << outSpatialSize <<"C" <<inSpatialSize << ") ";
  return inSpatialSize;
}
