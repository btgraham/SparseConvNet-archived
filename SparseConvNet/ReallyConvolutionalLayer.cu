// Performs a real convolution, used for the input layer. Other convolutions are
// implemented using ConvolutionLayer+NetworkInNetworkLayer

// 1) can use bz in dMultiply_Input_Weights_Output to access more
// output.nSpatialSites (4096*4096*32)?
// by+ty<outputNSpatialSites -> 4096*bz+by+ty<outputNSpatialSites, etc

#include "NetworkInNetworkLayer.h"
#include "ReallyConvolutionalLayer.h"
#include <iostream>
#include <vector>
#include <cassert>
#include "utilities.h"
#include "SigmoidLayer.h"
#include "Regions.h"

// Assume fs <=2*KERNELBLOCKSIZE

// // dMultiply_Input_Weights_Output <<<
// //
// dim3(output.featuresPresent.size()/KERNELBLOCKSIZE,(output.nSpatialSites+KERNELBLOCKSIZE-1)/KERNELBLOCKSIZE),
// //  dim3(KERNELBLOCKSIZE,KERNELBLOCKSIZE),0,memStream.stream>>>
// //
// (input.sub->features.dPtr(),w.dPtr(),b.dPtr(),output.rules.dPtr(),output.sub->features.dPtr(),
// //    input.featuresPresent.size(),output.featuresPresent.size(), fs,
// output.nSpatialSites,leaky);
__global__ void dMultiply_Input_Weights_Output(float *inFeatures, float *W,
                                               float *B, int *rules,
                                               float *outFeatures, int nIn,
                                               int nOut, int fs,
                                               int outputNSpatialSites,
                                               float leaky, float shrink = 1) {
  __shared__ float As[KERNELBLOCKSIZE][KERNELBLOCKSIZE];
  __shared__ float Bs[KERNELBLOCKSIZE][KERNELBLOCKSIZE];
  __shared__ int
      r[KERNELBLOCKSIZE][2 * KERNELBLOCKSIZE]; // Assume fs <=2*KERNELBLOCKSIZE
  int bx = blockIdx.x * KERNELBLOCKSIZE;
  int by = blockIdx.y * KERNELBLOCKSIZE;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  float acc = B[bx + tx];
  r[ty][tx] = (tx < fs and by + ty < outputNSpatialSites)
                  ? rules[(by + ty) * fs + tx]
                  : -1;
  r[ty][tx + KERNELBLOCKSIZE] =
      (tx + KERNELBLOCKSIZE < fs and by + ty < outputNSpatialSites)
          ? rules[(by + ty) * fs + tx + KERNELBLOCKSIZE]
          : -1;
  __syncthreads();
  for (int k = 0; k < nIn * fs; k += KERNELBLOCKSIZE) {
    int n = min(KERNELBLOCKSIZE, nIn * fs - k);
    int f = (k + tx) / nIn;
    int ff = (k + tx) % nIn;
    // int r=(tx<n and by+ty<outputNSpatialSites)?rules[(by+ty)*fs+f]:-1; /////
    // As[ty][tx]=(r>=0)?inFeatures[r*nIn+(ff)]:0; ////////
    As[ty][tx] = (r[ty][f] >= 0) ? inFeatures[r[ty][f] * nIn + (ff)] : 0;
    Bs[ty][tx] = (ty < n) ? W[(k + ty) * nOut + (bx + tx)] : 0;
    __syncthreads();
    for (int l = 0; l < n; l++)
      acc += As[ty][l] * Bs[l][tx];
    __syncthreads();
  }
  acc *= shrink;
  if (by + ty < outputNSpatialSites)
    outFeatures[(by + ty) * nOut + (bx + tx)] = (acc > 0) ? acc : (acc * leaky);
}

/************************************************************************************/
// // dMultiply_dOutput_WT_dInput
// //   <<<
// //
// dim3((input.featuresPresent.size()*fs+KERNELBLOCKSIZE-1)/KERNELBLOCKSIZE,(output.nSpatialSites+KERNELBLOCKSIZE-1)/KERNELBLOCKSIZE),
// //   dim3(KERNELBLOCKSIZE,KERNELBLOCKSIZE)
// //   ,0,memStream.stream>>>
__global__ void dMultiply_dOutput_WT_dInput(float *dOutFeatures, float *W,
                                            float *dInFeatures, int *rules,
                                            int nIn, int nOut, int fs,
                                            int outputNSpatialSites) {
  __shared__ float As[KERNELBLOCKSIZE][KERNELBLOCKSIZE];
  __shared__ float Bs[KERNELBLOCKSIZE][KERNELBLOCKSIZE];
  int bx = blockIdx.x * KERNELBLOCKSIZE;
  int by = blockIdx.y * KERNELBLOCKSIZE;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  float acc = 0;
  int n = min(KERNELBLOCKSIZE, nIn * fs - bx);
  for (int k = 0; k < nOut; k += KERNELBLOCKSIZE) {
    As[ty][tx] = (by + ty < outputNSpatialSites)
                     ? dOutFeatures[(by + ty) * nOut + k + tx]
                     : 0;
    Bs[tx][ty] = (ty < n) ? W[(bx + ty) * nOut + k + tx] : 0;
    __syncthreads();
    for (int l = 0; l < n; l++)
      acc += As[ty][l] * Bs[l][tx];
    __syncthreads();
  }
  int f = (bx + tx) / nIn;
  int ff = (bx + tx) % nIn;
  if (by + ty < outputNSpatialSites && f + tx < nIn * fs) {
    int r = rules[(by + ty) * fs + f];
    // dInFeatures[r*nIn+ff]+=acc;
    atomicAdd(&dInFeatures[r * nIn + ff], acc);
  }
}
/************************************************************************************/
// // dMultiply_InputT_dOutput_dWeights
// //   <<<
// //   dim3(output.featuresPresent.size()/KERNELBLOCKSIZE,
// //    (input.featuresPresent.size()*fs+KERNELBLOCKSIZE-1)/KERNELBLOCKSIZE,
// //    (output.nSpatialSites+KERNELBLOCKSIZE-1)/KERNELBLOCKSIZE),
// //   dim3(KERNELBLOCKSIZE,KERNELBLOCKSIZE)
// //   ,0,memStream.stream>>>
__global__ void dMultiply_InputT_dOutput_dWeights(float *inFeatures, int *rules,
                                                  float *dOutFeatures,
                                                  float *dW, int nIn, int nOut,
                                                  int fs,
                                                  int outputNSpatialSites) {
  __shared__ float As[KERNELBLOCKSIZE][KERNELBLOCKSIZE];
  __shared__ float Bs[KERNELBLOCKSIZE][KERNELBLOCKSIZE];
  int bx = blockIdx.x * KERNELBLOCKSIZE;
  int by = blockIdx.y * KERNELBLOCKSIZE;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  float acc = 0;
  int f = (by + ty) / nIn;
  int ff = (by + ty) % nIn;
  int k = blockIdx.z * KERNELBLOCKSIZE;
  {
    int n = min(KERNELBLOCKSIZE, outputNSpatialSites - k);
    int r = (tx < n and f < fs) ? rules[(k + tx) * fs + f] : -1;
    As[ty][tx] = (r >= 0) ? inFeatures[r * nIn + ff] : 0;
    Bs[ty][tx] = (ty < n) ? dOutFeatures[(k + ty) * nOut + bx + tx] : 0;
    __syncthreads();
    for (int l = 0; l < n; l++)
      acc += As[ty][l] * Bs[l][tx];
    __syncthreads();
  }
  if (f < fs)
    atomicAdd(&dW[(by + ty) * nOut + bx + tx], acc);
}

/************************************************************************************/
// Matrix is (nIn*fs)x(nOut)
// Shrink to (nInDropout*fs)*(nOut)
// inFeaturesPresent has length nInDropout
// outFeaturesPresent has length nOutDropout
__global__ void dShrinkMatrixForDropout(float *m, float *md,
                                        int *inFeaturesPresent,
                                        int *outFeaturesPresent, int nOut,
                                        int nOutDropout, int nIn,
                                        int nInDropout) {
  int i = blockIdx.x * nOutDropout;
  int ii = inFeaturesPresent[blockIdx.x % nInDropout] * nOut +
           (blockIdx.x / nInDropout) * nIn * nOut;
  for (int j = threadIdx.x; j < nOutDropout; j += KERNELBLOCKSIZE) {
    int jj = outFeaturesPresent[j];
    md[i + j] = m[ii + jj];
  }
}
__global__ void dGradientDescentShrunkMatrix(
    float *d_delta, float *d_momentum, float *d_weights, int nOut,
    int nOutDropout, int nIn, int nInDropout, int *inFeaturesPresent,
    int *outFeaturesPresent, float learningRate, float momentum) {
  int i = blockIdx.x * nOutDropout;
  int ii = inFeaturesPresent[blockIdx.x % nInDropout] * nOut +
           (blockIdx.x / nInDropout) * nIn * nOut;
  for (int j = threadIdx.x; j < nOutDropout; j += KERNELBLOCKSIZE) {
    int jj = outFeaturesPresent[j];
    // NAG light
    d_weights[ii + jj] -= d_momentum[ii + jj] * momentum;
    d_momentum[ii + jj] = momentum * d_momentum[ii + jj] -
                          learningRate * (1 - momentum) * d_delta[i + j];
    d_weights[ii + jj] =
        d_weights[ii + jj] + d_momentum[ii + jj] * (1 + momentum);
  }
}

ReallyConvolutionalLayer::ReallyConvolutionalLayer(
    cudaMemStream &memStream, int nFeaturesIn, int nFeaturesOut, int filterSize,
    int filterStride, int dimension, ActivationFunction fn, float dropout,
    int minActiveInputs, float poolingToFollow)
    : SpatiallySparseLayer(memStream), nFeaturesIn(nFeaturesIn),
      nFeaturesOut(nFeaturesOut), filterSize(filterSize),
      filterStride(filterStride), dimension(dimension), fn(fn),
      dropout(dropout), fs(ipow(filterSize, dimension)),
      W(true, nFeaturesIn * fs * nFeaturesOut),
      MW(true, nFeaturesIn * fs * nFeaturesOut), B(true, nFeaturesOut),
      MB(true, nFeaturesOut), minActiveInputs(minActiveInputs) {
  std::cout << "Convolution " << filterSize << "^" << dimension << "x"
            << nFeaturesIn << "=" << fs *nFeaturesIn << "->" << nFeaturesOut;
  if (filterStride > 1)
    std::cout << " stride " << filterStride;
  std::cout << " dropout=" << dropout << " " << sigmoidNames[fn] << std::endl;
  float scale = pow(
      6.0f / (nFeaturesIn * fs +
              nFeaturesOut *
                  powf(filterSize * 1.0 / filterStride / poolingToFollow, 2)),
      0.5f);
  W.setUniform(-scale, scale);
  MW.setZero();
  B.setZero();
  MB.setZero();
  switch (fn) {
  case RELU:
    leaky = 0;
    break;
  case LEAKYRELU:
    leaky = 0.01;
    break;
  case VLEAKYRELU:
    leaky = 0.333;
    break;
  default:
    assert(0);
  }
}
void ReallyConvolutionalLayer::preprocess(
    SpatiallySparseBatch &batch, SpatiallySparseBatchInterface &input,
    SpatiallySparseBatchInterface &output) {
  output.nFeatures = nFeaturesOut;
  assert(input.nFeatures == nFeaturesIn);
  assert(input.spatialSize >= filterSize);
  assert((input.spatialSize - filterSize) % filterStride == 0);
  output.spatialSize = (input.spatialSize - filterSize) / filterStride + 1;
  output.nSpatialSites = 0;
  output.grids.resize(batch.batchSize);
  output.backpropErrors = true;
  RegularSquareRegions regions(inSpatialSize, outSpatialSize, dimension,
                               filterSize, filterStride);
  for (int item = 0; item < batch.batchSize; item++) {
    gridRules(input.grids[item], output.grids[item], regions,
              output.nSpatialSites, output.rules.hVector(), minActiveInputs);
  }
  int o = nFeaturesOut * (batch.type == TRAINBATCH ? (1.0f - dropout) : 1.0f);
  output.featuresPresent.hVector() = rng.NchooseM(nFeaturesOut, o);
}
void ReallyConvolutionalLayer::forwards(SpatiallySparseBatch &batch,
                                        SpatiallySparseBatchInterface &input,
                                        SpatiallySparseBatchInterface &output) {
  output.sub->features.resize(output.nSpatialSites *
                              output.featuresPresent.size());
  if (batch.type == TRAINBATCH and
      nFeaturesIn + nFeaturesOut >
          input.featuresPresent.size() + output.featuresPresent.size()) {
    w.resize(input.featuresPresent.size() * fs * output.featuresPresent.size());
    dShrinkMatrixForDropout << <input.featuresPresent.size() * fs,
                                KERNELBLOCKSIZE, 0, memStream.stream>>>
        (W.dPtr(), w.dPtr(), input.featuresPresent.dPtr(),
         output.featuresPresent.dPtr(), output.nFeatures,
         output.featuresPresent.size(), input.nFeatures,
         input.featuresPresent.size());
    cudaCheckError();
    b.resize(output.featuresPresent.size());
    dShrinkVectorForDropout << <1, NTHREADS, 0, memStream.stream>>>
        (B.dPtr(), b.dPtr(), output.featuresPresent.dPtr(), output.nFeatures,
         output.featuresPresent.size());
    cudaCheckError();
    dMultiply_Input_Weights_Output
            << <dim3(output.featuresPresent.size() / KERNELBLOCKSIZE,
                     (output.nSpatialSites + KERNELBLOCKSIZE - 1) /
                         KERNELBLOCKSIZE),
                dim3(KERNELBLOCKSIZE, KERNELBLOCKSIZE), 0, memStream.stream>>>
        (input.sub->features.dPtr(), w.dPtr(), b.dPtr(), output.rules.dPtr(),
         output.sub->features.dPtr(), input.featuresPresent.size(),
         output.featuresPresent.size(), fs, output.nSpatialSites, leaky);
    cudaCheckError();
  } else {
    dMultiply_Input_Weights_Output
            << <dim3(output.featuresPresent.size() / KERNELBLOCKSIZE,
                     (output.nSpatialSites + KERNELBLOCKSIZE - 1) /
                         KERNELBLOCKSIZE),
                dim3(KERNELBLOCKSIZE, KERNELBLOCKSIZE), 0, memStream.stream>>>
        (input.sub->features.dPtr(), W.dPtr(), B.dPtr(), output.rules.dPtr(),
         output.sub->features.dPtr(), input.featuresPresent.size(),
         output.featuresPresent.size(), fs, output.nSpatialSites, leaky,
         1.0f - dropout);
    cudaCheckError();
  }
  multiplyAddCount += (__int128_t)output.nSpatialSites *
                      input.featuresPresent.size() * fs *
                      output.featuresPresent.size();
  cudaCheckError();
}
void ReallyConvolutionalLayer::scaleWeights(
    SpatiallySparseBatchInterface &input, SpatiallySparseBatchInterface &output,
    float &scalingUnderneath, bool topLayer) {
  assert(input.sub->features.size() > 0);
  assert(output.sub->features.size() > 0); // call after forwards(...)
  float scale = output.sub->features.meanAbs();
  std::cout << "featureScale:" << scale << std::endl;
  if (topLayer) {
    scale = 1;
  } else {
    scale = powf(
        scale,
        -0.1); // 0.7978846 = sqrt(2/pi) = mean of the half normal distribution
  }
  W.multiplicativeRescale(scale / scalingUnderneath);
  B.multiplicativeRescale(scale);
  MW.multiplicativeRescale(scale / scalingUnderneath);
  MB.multiplicativeRescale(scale);
  scalingUnderneath = scale;
}

void ReallyConvolutionalLayer::backwards(SpatiallySparseBatch &batch,
                                         SpatiallySparseBatchInterface &input,
                                         SpatiallySparseBatchInterface &output,
                                         float learningRate, float momentum) {
  applySigmoidBackProp(output, output, fn, memStream);
  dw.resize(input.featuresPresent.size() * fs * output.featuresPresent.size());
  dw.setZero(
      memStream); //////////////////////////////////////////////////////////////////
  db.resize(output.featuresPresent.size());
  db.setZero(memStream);
  columnSum(output.sub->dfeatures.dPtr(), db.dPtr(), output.nSpatialSites,
            output.featuresPresent.size(), memStream);
  cudaCheckError();
  dMultiply_InputT_dOutput_dWeights
          << <dim3(output.featuresPresent.size() / KERNELBLOCKSIZE,
                   (input.featuresPresent.size() * fs + KERNELBLOCKSIZE - 1) /
                       KERNELBLOCKSIZE,
                   (output.nSpatialSites + KERNELBLOCKSIZE - 1) /
                       KERNELBLOCKSIZE),
              dim3(KERNELBLOCKSIZE, KERNELBLOCKSIZE), 0, memStream.stream>>>
      (input.sub->features.dPtr(), output.rules.dPtr(),
       output.sub->dfeatures.dPtr(), dw.dPtr(), input.featuresPresent.size(),
       output.featuresPresent.size(), fs, output.nSpatialSites);
  multiplyAddCount += (__int128_t)output.nSpatialSites *
                      input.featuresPresent.size() * fs *
                      output.featuresPresent.size();
  cudaCheckError();

  if (nFeaturesIn + nFeaturesOut >
      input.featuresPresent.size() + output.featuresPresent.size()) {
    if (input.backpropErrors) {
      input.sub->dfeatures.resize(input.nSpatialSites *
                                  input.featuresPresent.size());
      input.sub->dfeatures.setZero(memStream);
      dMultiply_dOutput_WT_dInput << <
          dim3((input.featuresPresent.size() * fs + KERNELBLOCKSIZE - 1) /
                   KERNELBLOCKSIZE,
               (output.nSpatialSites + KERNELBLOCKSIZE - 1) / KERNELBLOCKSIZE),
          dim3(KERNELBLOCKSIZE, KERNELBLOCKSIZE), 0, memStream.stream>>>
          (output.sub->dfeatures.dPtr(), w.dPtr(), input.sub->dfeatures.dPtr(),
           output.rules.dPtr(), input.featuresPresent.size(),
           output.featuresPresent.size(), fs, output.nSpatialSites);
      multiplyAddCount += (__int128_t)output.nSpatialSites *
                          input.featuresPresent.size() * fs *
                          output.featuresPresent.size();
      cudaCheckError();
    }
    dGradientDescentShrunkMatrix << <input.featuresPresent.size(),
                                     KERNELBLOCKSIZE, 0, memStream.stream>>>
        (dw.dPtr(), MW.dPtr(), W.dPtr(), output.nFeatures,
         output.featuresPresent.size(), input.featuresPresent.dPtr(),
         output.featuresPresent.dPtr(), learningRate, momentum);

    dGradientDescentShrunkVector << <1, NTHREADS, 0, memStream.stream>>>
        (db.dPtr(), MB.dPtr(), B.dPtr(), output.nFeatures,
         output.featuresPresent.size(), output.featuresPresent.dPtr(),
         learningRate, momentum);
  } else {
    if (input.backpropErrors) {
      input.sub->dfeatures.resize(input.nSpatialSites *
                                  input.featuresPresent.size());
      input.sub->dfeatures.setZero(memStream);
      dMultiply_dOutput_WT_dInput << <
          dim3((input.featuresPresent.size() * fs + KERNELBLOCKSIZE - 1) /
                   KERNELBLOCKSIZE,
               (output.nSpatialSites + KERNELBLOCKSIZE - 1) / KERNELBLOCKSIZE),
          dim3(KERNELBLOCKSIZE, KERNELBLOCKSIZE), 0, memStream.stream>>>
          (output.sub->dfeatures.dPtr(), W.dPtr(), input.sub->dfeatures.dPtr(),
           output.rules.dPtr(), input.featuresPresent.size(),
           output.featuresPresent.size(), fs, output.nSpatialSites);
      multiplyAddCount += (__int128_t)output.nSpatialSites *
                          input.featuresPresent.size() * fs *
                          output.featuresPresent.size();
      cudaCheckError();
    }
    dGradientDescent << <nFeaturesIn, KERNELBLOCKSIZE, 0, memStream.stream>>>
        (dw.dPtr(), MW.dPtr(), W.dPtr(), nFeaturesOut, learningRate, momentum);
    dGradientDescent << <1, KERNELBLOCKSIZE, 0, memStream.stream>>>
        (db.dPtr(), MB.dPtr(), B.dPtr(), nFeaturesOut, learningRate, momentum);
  }
  cudaCheckError();
}
int ReallyConvolutionalLayer::calculateInputSpatialSize(int outputSpatialSize) {
  outSpatialSize = outputSpatialSize;
  inSpatialSize = filterSize + (outputSpatialSize - 1) * filterStride;
  std::cout << "-(C" << filterSize;
  if (filterStride != 1)
    std::cout << "/" << filterStride;
  std::cout << ")-" << inSpatialSize;
  return inSpatialSize;
}
