#include <iostream>
#include <cassert>
#include "utilities.h"
#include "MaxPoolingLayer.h"
#include "MaxPoolingTriangularLayer.h"
#include "Regions.h"

MaxPoolingTriangularLayer::MaxPoolingTriangularLayer(cudaMemStream &memStream,
                                                     int poolSize,
                                                     int poolStride,
                                                     int dimension)
    : SpatiallySparseLayer(memStream), poolSize(poolSize),
      poolStride(poolStride), dimension(dimension) {
  S = triangleSize(poolSize, dimension);
  std::cout << dimension
            << "D MaxPoolingTriangularLayer side-length=" << poolSize
            << " volume=" << S << " stride " << poolStride << std::endl;
}
void MaxPoolingTriangularLayer::preprocess(
    SpatiallySparseBatch &batch, SpatiallySparseBatchInterface &input,
    SpatiallySparseBatchInterface &output) {
  assert(input.spatialSize == inSpatialSize);
  output.nFeatures = input.nFeatures;
  output.featuresPresent.hVector() = input.featuresPresent.hVector();
  output.spatialSize = outSpatialSize;
  output.nSpatialSites = 0;
  output.grids.resize(batch.batchSize);
  output.backpropErrors = input.backpropErrors;
  RegularTriangularRegions regions(inSpatialSize, outSpatialSize, dimension,
                                   poolSize, poolStride);
  for (int item = 0; item < batch.batchSize; item++)
    gridRules(input.grids[item], output.grids[item], regions,
              output.nSpatialSites, output.rules.hVector());
}
void MaxPoolingTriangularLayer::forwards(
    SpatiallySparseBatch &batch, SpatiallySparseBatchInterface &input,
    SpatiallySparseBatchInterface &output) {
  output.sub->poolingChoices.resize(output.nSpatialSites *
                                    output.featuresPresent.size());
  output.sub->features.resize(output.nSpatialSites *
                              output.featuresPresent.size());
  cudaCheckError();
  maxPool(input.sub->features.dPtr(), output.sub->features.dPtr(),
          output.rules.dPtr(), output.nSpatialSites, S,
          output.featuresPresent.size(), output.sub->poolingChoices.dPtr(),
          memStream);
  cudaCheckError();
}
void MaxPoolingTriangularLayer::backwards(SpatiallySparseBatch &batch,
                                          SpatiallySparseBatchInterface &input,
                                          SpatiallySparseBatchInterface &output,
                                          float learningRate, float momentum) {
  if (input.backpropErrors) {
    input.sub->dfeatures.resize(input.nSpatialSites *
                                input.featuresPresent.size());
    input.sub->dfeatures.setZero(memStream);
    maxPoolBackProp(input.sub->dfeatures.dPtr(), output.sub->dfeatures.dPtr(),
                    output.nSpatialSites, output.featuresPresent.size(),
                    output.sub->poolingChoices.dPtr(), memStream);
  }
}
int MaxPoolingTriangularLayer::calculateInputSpatialSize(
    int outputSpatialSize) {
  outSpatialSize = outputSpatialSize;
  inSpatialSize = poolSize + (outputSpatialSize - 1) * poolStride;
  std::cout << "-(MP" << poolSize;
  if (poolStride != poolSize)
    std::cout << "/" << poolStride;
  std::cout << ")-" << inSpatialSize;
  return inSpatialSize;
}
