#include "ConvolutionalTriangularLayer.h"
#include "ConvolutionalLayer.h"
#include <iostream>
#include <cassert>
#include "utilities.h"
#include "Regions.h"

ConvolutionalTriangularLayer::ConvolutionalTriangularLayer(
    cudaMemStream &memStream, int filterSize, int filterStride, int dimension,
    int nFeaturesIn, int minActiveInputs)
    : SpatiallySparseLayer(memStream), filterSize(filterSize),
      filterStride(filterStride), dimension(dimension),
      nFeaturesIn(nFeaturesIn), minActiveInputs(minActiveInputs) {
  fs = triangleSize(filterSize, dimension);
  nFeaturesOut = fs * nFeaturesIn;
  std::cout << dimension
            << "D ConvolutionalTriangularLayer side-length=" << filterSize
            << " " << nFeaturesIn << "x" << fs << "->" << nFeaturesOut;
  if (filterStride > 1)
    std::cout << ", stride " << filterStride;
  std::cout << std::endl;
}
void ConvolutionalTriangularLayer::preprocess(
    SpatiallySparseBatch &batch, SpatiallySparseBatchInterface &input,
    SpatiallySparseBatchInterface &output) {
  assert(input.nFeatures == nFeaturesIn);
  assert(input.spatialSize >= filterSize);
  assert((input.spatialSize - filterSize) % filterStride == 0);
  output.nFeatures = nFeaturesOut;
  output.spatialSize = (input.spatialSize - filterSize) / filterStride + 1;
  output.nSpatialSites = 0;
  output.grids.resize(batch.batchSize);
  output.backpropErrors = input.backpropErrors;
  RegularTriangularRegions regions(inSpatialSize, outSpatialSize, dimension,
                                   filterSize, filterStride);
  for (int item = 0; item < batch.batchSize; item++)
    gridRules(input.grids[item], output.grids[item], regions,
              output.nSpatialSites, output.rules.hVector(), minActiveInputs);
  output.featuresPresent.resize(input.featuresPresent.size() * fs);
  convolutionFeaturesPresent(input.featuresPresent.hVector(),
                             output.featuresPresent.hVector(), input.nFeatures,
                             input.featuresPresent.size(), fs);
}
void ConvolutionalTriangularLayer::forwards(
    SpatiallySparseBatch &batch, SpatiallySparseBatchInterface &input,
    SpatiallySparseBatchInterface &output) {
  output.sub->features.resize(output.nSpatialSites *
                              output.featuresPresent.size());
  propForwardToMatrixMultiply(input.sub->features.dPtr(),
                              output.sub->features.dPtr(), output.rules.dPtr(),
                              output.nSpatialSites * fs,
                              input.featuresPresent.size(), memStream);
}
void ConvolutionalTriangularLayer::backwards(
    SpatiallySparseBatch &batch, SpatiallySparseBatchInterface &input,
    SpatiallySparseBatchInterface &output, float learningRate, float momentum) {
  if (input.backpropErrors) {
    input.sub->dfeatures.resize(input.nSpatialSites *
                                input.featuresPresent.size());
    input.sub->dfeatures.setZero(memStream);
    propBackwardFromMatrixMultiply(
        input.sub->dfeatures.dPtr(), output.sub->dfeatures.dPtr(),
        output.rules.dPtr(), output.nSpatialSites * fs,
        input.featuresPresent.size(), memStream);
  }
}
int ConvolutionalTriangularLayer::calculateInputSpatialSize(
    int outputSpatialSize) {
  outSpatialSize = outputSpatialSize;
  inSpatialSize = filterSize + (outputSpatialSize - 1) * filterStride;
  std::cout << "-(C" << filterSize;
  if (filterStride != 1)
    std::cout << "/" << filterStride;
  std::cout << ")-" << inSpatialSize;
  return inSpatialSize;
}
