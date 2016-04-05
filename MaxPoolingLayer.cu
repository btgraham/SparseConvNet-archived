#include <iostream>
#include <cassert>
#include "utilities.h"
#include "MaxPoolingLayer.h"
#include "Regions.h"

// Values of -1 in "rules" used to indicate pooling region is smaller than "ps"
// Call with maxPS >= ps
template <int maxPS>
__global__ void dMaxPool(float *g1, float *g2, int *rules, int nOut,
                         int *d_choice, int ps) {
  __shared__ int r[maxPS];
  int i = blockIdx.x * nOut; // for output
  for (int p = threadIdx.x; p < ps; p += KERNELBLOCKSIZE)
    r[p] = rules[blockIdx.x * ps + p] * nOut; // for input
  __syncthreads();
  for (int j = threadIdx.x; j < nOut; j += KERNELBLOCKSIZE) {
    bool notFoundAnyPositive_rp = true;
    float t;
    int c;
    for (int p = 0; p < ps; p++) {
      if (r[p] >= 0) {
        float s = (r[p] >= 0) ? g1[r[p] + j] : -10000000;
        if (notFoundAnyPositive_rp or t < s) {
          notFoundAnyPositive_rp = false;
          c = r[p] + j;
          t = s;
        }
      }
    }
    g2[i + j] = t;
    d_choice[i + j] = c;
    __syncthreads();
  }
}

void maxPool(float *g1, float *g2, int *rules, int count, int sd, int nOut,
             int *d_choice, cudaMemStream &memStream) {
  // std::cout << g1 << " " << g2 << " " << rules << " " <<count << " " << sd <<
  // " " << nOut << " " << d_choice << std::endl;
  int processed = 0;
  while (processed < count) {
    int batch = min(32768, count - processed);
    if (sd <= 8) {
      dMaxPool<8> << <batch, KERNELBLOCKSIZE, 0, memStream.stream>>>
          (g1, g2 + processed * nOut, rules + processed * sd, nOut,
           d_choice + processed * nOut, sd);
    } else if (sd <= 16) {
      dMaxPool<16> << <batch, KERNELBLOCKSIZE, 0, memStream.stream>>>
          (g1, g2 + processed * nOut, rules + processed * sd, nOut,
           d_choice + processed * nOut, sd);
    } else if (sd <= 32) {
      dMaxPool<32> << <batch, KERNELBLOCKSIZE, 0, memStream.stream>>>
          (g1, g2 + processed * nOut, rules + processed * sd, nOut,
           d_choice + processed * nOut, sd);
    } else if (sd <= 64) {
      dMaxPool<64> << <batch, KERNELBLOCKSIZE, 0, memStream.stream>>>
          (g1, g2 + processed * nOut, rules + processed * sd, nOut,
           d_choice + processed * nOut, sd);
    } else if (sd <= 1024) {
      dMaxPool<1024> << <batch, KERNELBLOCKSIZE, 0, memStream.stream>>>
          (g1, g2 + processed * nOut, rules + processed * sd, nOut,
           d_choice + processed * nOut, sd);
    } else {
      std::cout << "Do some copying and pasting in " << __FILE__ << " line "
                << __LINE__ << " sd=" << sd << std::endl;
      exit(1);
    }
    processed += batch;
  }
  cudaCheckError();
}

__global__ void dMaxPoolBackProp(float *d1, float *d2, int nOut,
                                 int *d_choice) {
  // initialize d1 to zero first!!!
  int i = blockIdx.x * nOut;
  for (int j = i + threadIdx.x; j < i + nOut; j += KERNELBLOCKSIZE) {
    atomicAdd(&d1[d_choice[j]], d2[j]); // d1[d_choice[j]]=d2[j];
  }
}

void maxPoolBackProp(float *d1, float *d2, int count, int nOut, int *d_choice,
                     cudaMemStream &memStream) {
  int processed = 0;
  while (processed < count) {
    int batch = min(32768, count - processed);
    dMaxPoolBackProp << <batch, KERNELBLOCKSIZE, 0, memStream.stream>>>
        (d1, d2 + processed * nOut, nOut, d_choice + processed * nOut);
    processed += batch;
  }
  cudaCheckError();
}

// TODO: Refactor the different pooling classes somehow

MaxPoolingLayer::MaxPoolingLayer(cudaMemStream &memStream, int poolSize,
                                 int poolStride, int dimension)
    : SpatiallySparseLayer(memStream), poolSize(poolSize),
      poolStride(poolStride), dimension(dimension) {
  sd = ipow(poolSize, dimension);
  std::cout << "MaxPooling " << poolSize << " " << poolStride << std::endl;
}
void MaxPoolingLayer::preprocess(SpatiallySparseBatch &batch,
                                 SpatiallySparseBatchInterface &input,
                                 SpatiallySparseBatchInterface &output) {
  assert(input.spatialSize == inSpatialSize);
  output.nFeatures = input.nFeatures;
  output.featuresPresent.hVector() = input.featuresPresent.hVector();
  output.spatialSize = outSpatialSize;
  output.nSpatialSites = 0;
  output.grids.resize(batch.batchSize);
  output.backpropErrors = input.backpropErrors;
  RegularSquareRegions regions(inSpatialSize, outSpatialSize, dimension,
                               poolSize, poolStride);
  for (int item = 0; item < batch.batchSize; item++)
    gridRules(input.grids[item], output.grids[item], regions,
              output.nSpatialSites, output.rules.hVector(), true);
}
void MaxPoolingLayer::forwards(SpatiallySparseBatch &batch,
                               SpatiallySparseBatchInterface &input,
                               SpatiallySparseBatchInterface &output) {
  output.sub->poolingChoices.resize(output.nSpatialSites *
                                    output.featuresPresent.size());
  output.sub->features.resize(output.nSpatialSites *
                              output.featuresPresent.size());
  cudaCheckError();
  maxPool(input.sub->features.dPtr(), output.sub->features.dPtr(),
          output.rules.dPtr(), output.nSpatialSites, sd,
          output.featuresPresent.size(), output.sub->poolingChoices.dPtr(),
          memStream);
  cudaCheckError();
}
void MaxPoolingLayer::backwards(SpatiallySparseBatch &batch,
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
int MaxPoolingLayer::calculateInputSpatialSize(int outputSpatialSize) {
  outSpatialSize = outputSpatialSize;
  inSpatialSize = poolSize + (outputSpatialSize - 1) * poolStride;
  std::cout << "-(MP" << poolSize;
  if (poolStride != poolSize)
    std::cout << "/" << poolStride;
  std::cout << ")-" << inSpatialSize;
  return inSpatialSize;
}

PseudorandomOverlappingFractionalMaxPoolingLayer::
    PseudorandomOverlappingFractionalMaxPoolingLayer(cudaMemStream &memStream,
                                                     int poolSize,
                                                     float fmpShrink,
                                                     int dimension)
    : SpatiallySparseLayer(memStream), poolSize(poolSize), fmpShrink(fmpShrink),
      dimension(dimension) {
  sd = ipow(poolSize, dimension);
  std::cout << "Pseudorandom overlapping Fractional Max Pooling " << fmpShrink
            << " " << poolSize << std::endl;
}
void PseudorandomOverlappingFractionalMaxPoolingLayer::preprocess(
    SpatiallySparseBatch &batch, SpatiallySparseBatchInterface &input,
    SpatiallySparseBatchInterface &output) {
  assert(input.spatialSize == inSpatialSize);
  output.nFeatures = input.nFeatures;
  output.featuresPresent.hVector() = input.featuresPresent.hVector();
  output.spatialSize = outSpatialSize;
  output.nSpatialSites = 0;
  output.grids.resize(batch.batchSize);
  output.backpropErrors = input.backpropErrors;
  for (int item = 0; item < batch.batchSize; item++) {
    FractionalPoolingRegions<PseudorandomOverlappingFmpTicks> regions(
        inSpatialSize, outSpatialSize, dimension, poolSize, rng);
    gridRules(input.grids[item], output.grids[item], regions,
              output.nSpatialSites, output.rules.hVector(), true);
  }
}
void PseudorandomOverlappingFractionalMaxPoolingLayer::forwards(
    SpatiallySparseBatch &batch, SpatiallySparseBatchInterface &input,
    SpatiallySparseBatchInterface &output) {
  output.sub->poolingChoices.resize(output.nSpatialSites *
                                    output.featuresPresent.size());
  output.sub->features.resize(output.nSpatialSites *
                              output.featuresPresent.size());
  cudaCheckError();
  maxPool(input.sub->features.dPtr(), output.sub->features.dPtr(),
          output.rules.dPtr(), output.nSpatialSites, sd,
          output.featuresPresent.size(), output.sub->poolingChoices.dPtr(),
          memStream);
  cudaCheckError();
}
void PseudorandomOverlappingFractionalMaxPoolingLayer::backwards(
    SpatiallySparseBatch &batch, SpatiallySparseBatchInterface &input,
    SpatiallySparseBatchInterface &output, float learningRate, float momentum) {
  if (input.backpropErrors) {
    input.sub->dfeatures.resize(input.nSpatialSites *
                                input.featuresPresent.size());
    input.sub->dfeatures.setZero(memStream);
    maxPoolBackProp(input.sub->dfeatures.dPtr(), output.sub->dfeatures.dPtr(),
                    output.nSpatialSites, output.featuresPresent.size(),
                    output.sub->poolingChoices.dPtr(), memStream);
  }
}
int PseudorandomOverlappingFractionalMaxPoolingLayer::calculateInputSpatialSize(
    int outputSpatialSize) {
  outSpatialSize = outputSpatialSize;
  inSpatialSize = outputSpatialSize * fmpShrink + 0.5;
  if (inSpatialSize == outputSpatialSize)
    inSpatialSize++;
  std::cout << "-(POFMP)-" << inSpatialSize;
  return inSpatialSize;
}

RandomOverlappingFractionalMaxPoolingLayer::
    RandomOverlappingFractionalMaxPoolingLayer(cudaMemStream &memStream,
                                               int poolSize, float fmpShrink,
                                               int dimension)
    : SpatiallySparseLayer(memStream), poolSize(poolSize), fmpShrink(fmpShrink),
      dimension(dimension) {
  sd = ipow(poolSize, dimension);
  std::cout << "Random overlapping Fractional Max Pooling " << fmpShrink << " "
            << poolSize << std::endl;
}
void RandomOverlappingFractionalMaxPoolingLayer::preprocess(
    SpatiallySparseBatch &batch, SpatiallySparseBatchInterface &input,
    SpatiallySparseBatchInterface &output) {
  assert(input.spatialSize == inSpatialSize);
  output.nFeatures = input.nFeatures;
  output.featuresPresent.hVector() = input.featuresPresent.hVector();
  output.spatialSize = outSpatialSize;
  output.nSpatialSites = 0;
  output.grids.resize(batch.batchSize);
  output.backpropErrors = input.backpropErrors;
  for (int item = 0; item < batch.batchSize; item++) {
    FractionalPoolingRegions<RandomOverlappingFmpTicks> regions(
        inSpatialSize, outSpatialSize, dimension, poolSize, rng);
    gridRules(input.grids[item], output.grids[item], regions,
              output.nSpatialSites, output.rules.hVector(), true);
  }
}
void RandomOverlappingFractionalMaxPoolingLayer::forwards(
    SpatiallySparseBatch &batch, SpatiallySparseBatchInterface &input,
    SpatiallySparseBatchInterface &output) {
  output.sub->poolingChoices.resize(output.nSpatialSites *
                                    output.featuresPresent.size());
  output.sub->features.resize(output.nSpatialSites *
                              output.featuresPresent.size());
  cudaCheckError();
  maxPool(input.sub->features.dPtr(), output.sub->features.dPtr(),
          output.rules.dPtr(), output.nSpatialSites, sd,
          output.featuresPresent.size(), output.sub->poolingChoices.dPtr(),
          memStream);
  cudaCheckError();
}
void RandomOverlappingFractionalMaxPoolingLayer::backwards(
    SpatiallySparseBatch &batch, SpatiallySparseBatchInterface &input,
    SpatiallySparseBatchInterface &output, float learningRate, float momentum) {
  if (input.backpropErrors) {
    input.sub->dfeatures.resize(input.nSpatialSites *
                                input.featuresPresent.size());
    input.sub->dfeatures.setZero(memStream);
    maxPoolBackProp(input.sub->dfeatures.dPtr(), output.sub->dfeatures.dPtr(),
                    output.nSpatialSites, output.featuresPresent.size(),
                    output.sub->poolingChoices.dPtr(), memStream);
  }
}
int RandomOverlappingFractionalMaxPoolingLayer::calculateInputSpatialSize(
    int outputSpatialSize) {
  outSpatialSize = outputSpatialSize;
  inSpatialSize = outputSpatialSize * fmpShrink + 0.5;
  if (inSpatialSize == outputSpatialSize)
    inSpatialSize++;
  std::cout << "-(ROFMP)-" << inSpatialSize;
  return inSpatialSize;
}

PseudorandomNonOverlappingFractionalMaxPoolingLayer::
    PseudorandomNonOverlappingFractionalMaxPoolingLayer(
        cudaMemStream &memStream, int poolSize, float fmpShrink, int dimension)
    : SpatiallySparseLayer(memStream), poolSize(poolSize), fmpShrink(fmpShrink),
      dimension(dimension) {
  sd = ipow(poolSize, dimension);
  std::cout << "Pseudorandom non-overlapping Fractional Max Pooling "
            << fmpShrink << " " << poolSize << std::endl;
}
void PseudorandomNonOverlappingFractionalMaxPoolingLayer::preprocess(
    SpatiallySparseBatch &batch, SpatiallySparseBatchInterface &input,
    SpatiallySparseBatchInterface &output) {
  assert(input.spatialSize == inSpatialSize);
  output.nFeatures = input.nFeatures;
  output.featuresPresent.hVector() = input.featuresPresent.hVector();
  output.spatialSize = outSpatialSize;
  output.nSpatialSites = 0;
  output.grids.resize(batch.batchSize);
  output.backpropErrors = input.backpropErrors;
  for (int item = 0; item < batch.batchSize; item++) {
    FractionalPoolingRegions<PseudorandomNonOverlappingFmpTicks> regions(
        inSpatialSize, outSpatialSize, dimension, poolSize, rng);
    gridRules(input.grids[item], output.grids[item], regions,
              output.nSpatialSites, output.rules.hVector(), false);
  }
}
void PseudorandomNonOverlappingFractionalMaxPoolingLayer::forwards(
    SpatiallySparseBatch &batch, SpatiallySparseBatchInterface &input,
    SpatiallySparseBatchInterface &output) {
  output.sub->poolingChoices.resize(output.nSpatialSites *
                                    output.featuresPresent.size());
  output.sub->features.resize(output.nSpatialSites *
                              output.featuresPresent.size());
  cudaCheckError();
  maxPool(input.sub->features.dPtr(), output.sub->features.dPtr(),
          output.rules.dPtr(), output.nSpatialSites, sd,
          output.featuresPresent.size(), output.sub->poolingChoices.dPtr(),
          memStream);
  cudaCheckError();
}
void PseudorandomNonOverlappingFractionalMaxPoolingLayer::backwards(
    SpatiallySparseBatch &batch, SpatiallySparseBatchInterface &input,
    SpatiallySparseBatchInterface &output, float learningRate, float momentum) {
  if (input.backpropErrors) {
    input.sub->dfeatures.resize(input.nSpatialSites *
                                input.featuresPresent.size());
    input.sub->dfeatures.setZero(memStream);
    maxPoolBackProp(input.sub->dfeatures.dPtr(), output.sub->dfeatures.dPtr(),
                    output.nSpatialSites, output.featuresPresent.size(),
                    output.sub->poolingChoices.dPtr(), memStream);
  }
}
int PseudorandomNonOverlappingFractionalMaxPoolingLayer::
    calculateInputSpatialSize(int outputSpatialSize) {
  outSpatialSize = outputSpatialSize;
  inSpatialSize = outputSpatialSize * fmpShrink + 0.5;
  if (inSpatialSize == outputSpatialSize)
    inSpatialSize++;
  std::cout << "-(PDFMP)-" << inSpatialSize;
  return inSpatialSize;
}

RandomNonOverlappingFractionalMaxPoolingLayer::
    RandomNonOverlappingFractionalMaxPoolingLayer(cudaMemStream &memStream,
                                                  int poolSize, float fmpShrink,
                                                  int dimension)
    : SpatiallySparseLayer(memStream), poolSize(poolSize), fmpShrink(fmpShrink),
      dimension(dimension) {
  sd = ipow(poolSize, dimension);
  std::cout << "Random non-overlapping Fractional Max Pooling " << fmpShrink
            << " " << poolSize << std::endl;
}
void RandomNonOverlappingFractionalMaxPoolingLayer::preprocess(
    SpatiallySparseBatch &batch, SpatiallySparseBatchInterface &input,
    SpatiallySparseBatchInterface &output) {
  assert(input.spatialSize == inSpatialSize);
  output.nFeatures = input.nFeatures;
  output.featuresPresent.hVector() = input.featuresPresent.hVector();
  output.spatialSize = outSpatialSize;
  output.nSpatialSites = 0;
  output.grids.resize(batch.batchSize);
  output.backpropErrors = input.backpropErrors;
  for (int item = 0; item < batch.batchSize; item++) {
    FractionalPoolingRegions<RandomNonOverlappingFmpTicks> regions(
        inSpatialSize, outSpatialSize, dimension, poolSize, rng);
    gridRules(input.grids[item], output.grids[item], regions,
              output.nSpatialSites, output.rules.hVector(), false);
  }
}
void RandomNonOverlappingFractionalMaxPoolingLayer::forwards(
    SpatiallySparseBatch &batch, SpatiallySparseBatchInterface &input,
    SpatiallySparseBatchInterface &output) {
  output.sub->poolingChoices.resize(output.nSpatialSites *
                                    output.featuresPresent.size());
  output.sub->features.resize(output.nSpatialSites *
                              output.featuresPresent.size());
  cudaCheckError();
  maxPool(input.sub->features.dPtr(), output.sub->features.dPtr(),
          output.rules.dPtr(), output.nSpatialSites, sd,
          output.featuresPresent.size(), output.sub->poolingChoices.dPtr(),
          memStream);
  cudaCheckError();
}
void RandomNonOverlappingFractionalMaxPoolingLayer::backwards(
    SpatiallySparseBatch &batch, SpatiallySparseBatchInterface &input,
    SpatiallySparseBatchInterface &output, float learningRate, float momentum) {
  if (input.backpropErrors) {
    input.sub->dfeatures.resize(input.nSpatialSites *
                                input.featuresPresent.size());
    input.sub->dfeatures.setZero(memStream);
    maxPoolBackProp(input.sub->dfeatures.dPtr(), output.sub->dfeatures.dPtr(),
                    output.nSpatialSites, output.featuresPresent.size(),
                    output.sub->poolingChoices.dPtr(), memStream);
  }
}
int RandomNonOverlappingFractionalMaxPoolingLayer::calculateInputSpatialSize(
    int outputSpatialSize) {
  outSpatialSize = outputSpatialSize;
  inSpatialSize = outputSpatialSize * fmpShrink + 0.5;
  if (inSpatialSize == outputSpatialSize)
    inSpatialSize++;
  std::cout << "-(RDFMP)-" << inSpatialSize;
  return inSpatialSize;
}
