#include "NetworkInNetworkLayer.h"
#include "utilities.h"
#include "SigmoidLayer.h"
#include <iostream>
#include <cassert>

__global__ void dShrinkMatrixForDropout(float *m, float *md,
                                        int *inFeaturesPresent,
                                        int *outFeaturesPresent, int nOut,
                                        int nOutDropout) {
  int i = blockIdx.x * nOutDropout;
  int ii = inFeaturesPresent[blockIdx.x] * nOut;
  for (int j = threadIdx.x; j < nOutDropout; j += KERNELBLOCKSIZE) {
    int jj = outFeaturesPresent[j];
    md[i + j] = m[ii + jj];
  }
}

__global__ void dShrinkVectorForDropout(float *m, float *md,
                                        int *outFeaturesPresent, int nOut,
                                        int nOutDropout) {
  for (int i = threadIdx.x; i < nOutDropout; i += NTHREADS) {
    md[i] = m[outFeaturesPresent[i]];
  }
}

__global__ void dGradientDescent(float *d_delta, float *d_momentum,
                                 float *d_weights, int nOut, float learningRate,
                                 float momentum) {
  int i = blockIdx.x * nOut;
  for (int j = i + threadIdx.x; j < i + nOut; j += KERNELBLOCKSIZE) {
    float w = d_weights[j];
    float m = d_momentum[j];
    float d = d_delta[j];
    w -= m * momentum;
    m = momentum * m - learningRate * (1 - momentum) * d;
    w += m * (1 + momentum);
    d_weights[j] = w;
    d_momentum[j] = m;
  }
}

__global__ void
dGradientDescentShrunkMatrix(float *d_delta, float *d_momentum,
                             float *d_weights, int nOut, int nOutDropout,
                             int *inFeaturesPresent, int *outFeaturesPresent,
                             float learningRate, float momentum) {
  int i = blockIdx.x * nOutDropout;
  int ii = inFeaturesPresent[blockIdx.x] * nOut;
  for (int j = threadIdx.x; j < nOutDropout; j += KERNELBLOCKSIZE) {
    int jj = outFeaturesPresent[j];
    // NAG light
    float w = d_weights[ii + jj];
    float m = d_momentum[ii + jj];
    float d = d_delta[i + j];
    w -= m * momentum;
    m = momentum * m - learningRate * (1 - momentum) * d;
    w += m * (1 + momentum);
    d_weights[ii + jj] = w;
    d_momentum[ii + jj] = m;
  }
}

__global__ void dGradientDescentShrunkVector(float *d_delta, float *d_momentum,
                                             float *d_weights, int nOut,
                                             int nOutDropout,
                                             int *outFeaturesPresent,
                                             float learningRate,
                                             float momentum) {
  for (int i = threadIdx.x; i < nOutDropout; i += NTHREADS) {
    int ii = outFeaturesPresent[i];
    // NAG light
    float w = d_weights[ii];
    float m = d_momentum[ii];
    float d = d_delta[i];
    w -= m * momentum;
    m = momentum * m - learningRate * (1 - momentum) * d;
    w += m * (1 + momentum);
    d_weights[ii] = w;
    d_momentum[ii] = m;
  }
}

__global__ void dColumnSum(float *matrix, float *target, int nRows,
                           int nColumns) {
  int i = blockIdx.x * KERNELBLOCKSIZE + threadIdx.x;
  float t = 0;
  for (int j = blockIdx.y; j < nRows; j += KERNELBLOCKSIZE)
    t += matrix[j * nColumns + i];
  atomicAdd(&target[i], t);
}
void columnSum(float *matrix, float *target, int nRows, int nColumns,
               cudaMemStream &memStream) {
  if (nColumns / KERNELBLOCKSIZE > 0)
    dColumnSum << <dim3(nColumns / KERNELBLOCKSIZE, KERNELBLOCKSIZE),
                   KERNELBLOCKSIZE, 0, memStream.stream>>>
        (matrix, target, nRows, nColumns);
  if (nColumns % KERNELBLOCKSIZE > 0) {
    int o = nColumns / KERNELBLOCKSIZE * KERNELBLOCKSIZE;
    dColumnSum
            << <dim3(1, KERNELBLOCKSIZE), nColumns - o, 0, memStream.stream>>>
        (matrix + o, target + o, nRows, nColumns);
  }
  cudaCheckError();
}

__global__ void dReplicateArray(float *src, float *dst, int nColumns) {
  int i = blockIdx.x * nColumns;
  for (int j = threadIdx.x; j < nColumns; j += KERNELBLOCKSIZE)
    dst[i + j] = src[j];
}
void replicateArray(float *src, float *dst, int nRows, int nColumns,
                    cudaMemStream &memStream) {
  int processed = 0;
  while (processed < nRows) {
    int batch =
        min(1024, nRows - processed); //////////////////////////////////////
    dReplicateArray << <batch, KERNELBLOCKSIZE, 0, memStream.stream>>>
        (src, dst + processed * nColumns, nColumns);
    processed += batch;
  }
  cudaCheckError();
}

NetworkInNetworkLayer::NetworkInNetworkLayer(
    cudaMemStream &memStream, cublasHandle_t &cublasHandle, int nFeaturesIn,
    int nFeaturesOut, float dropout, ActivationFunction fn,
    float alpha // used to determine intialization weights only
    )
    : SpatiallySparseLayer(memStream), cublasHandle(cublasHandle),
      nFeaturesIn(nFeaturesIn), nFeaturesOut(nFeaturesOut), dropout(dropout),
      fn(fn), W(true, nFeaturesIn * nFeaturesOut),
      MW(true, nFeaturesIn * nFeaturesOut), B(true, nFeaturesOut),
      MB(true, nFeaturesOut) {
  float scale = pow(6.0f / (nFeaturesIn + nFeaturesOut * alpha), 0.5f);
  W.copyToCPUAsync(memStream);
  W.setUniform(-scale, scale);
  W.copyToGPUAsync(memStream);
  MW.setZero();
  B.setZero();
  MB.setZero();
  std::cout << "Learn " << nFeaturesIn << "->" << nFeaturesOut
            << " dropout=" << dropout << " " << sigmoidNames[fn] << std::endl;
}
void NetworkInNetworkLayer::preprocess(SpatiallySparseBatch &batch,
                                       SpatiallySparseBatchInterface &input,
                                       SpatiallySparseBatchInterface &output) {
  assert(input.nFeatures == nFeaturesIn);
  output.nFeatures = nFeaturesOut;
  output.spatialSize = input.spatialSize;
  output.nSpatialSites = input.nSpatialSites;
  output.grids = input.grids;
  int o = nFeaturesOut * (batch.type == TRAINBATCH ? (1.0f - dropout) : 1.0f);
  output.featuresPresent.hVector() = rng.NchooseM(nFeaturesOut, o);
  output.backpropErrors = true;
}
void NetworkInNetworkLayer::forwards(SpatiallySparseBatch &batch,
                                     SpatiallySparseBatchInterface &input,
                                     SpatiallySparseBatchInterface &output) {
  output.sub->features.resize(output.nSpatialSites *
                              output.featuresPresent.size());
  if (batch.type == TRAINBATCH and
      nFeaturesIn + nFeaturesOut >
          input.featuresPresent.size() + output.featuresPresent.size()) {
    w.resize(input.featuresPresent.size() * output.featuresPresent.size());
    dShrinkMatrixForDropout << <input.featuresPresent.size(), KERNELBLOCKSIZE,
                                0, memStream.stream>>>
        (W.dPtr(), w.dPtr(), input.featuresPresent.dPtr(),
         output.featuresPresent.dPtr(), output.nFeatures,
         output.featuresPresent.size());
    cudaCheckError();
    b.resize(output.featuresPresent.size());
    dShrinkVectorForDropout << <1, NTHREADS, 0, memStream.stream>>>
        (B.dPtr(), b.dPtr(), output.featuresPresent.dPtr(), output.nFeatures,
         output.featuresPresent.size());
    cudaCheckError();
    replicateArray(b.dPtr(), output.sub->features.dPtr(), output.nSpatialSites,
                   output.featuresPresent.size(), memStream);
    cudaCheckError();
    d_rowMajorSGEMM_alphaAB_betaC(
        cublasHandle, input.sub->features.dPtr(), w.dPtr(),
        output.sub->features.dPtr(), output.nSpatialSites,
        input.featuresPresent.size(), output.featuresPresent.size(), 1.0f, 1.0f,
        __FILE__, __LINE__);
    cudaCheckError();

  } else {
    float p = 1.0f - (batch.type != RESCALEBATCH ? dropout : 0.0f);
    replicateArray(B.dPtr(), output.sub->features.dPtr(), output.nSpatialSites,
                   output.featuresPresent.size(), memStream);
    d_rowMajorSGEMM_alphaAB_betaC(cublasHandle, input.sub->features.dPtr(),
                                  W.dPtr(), output.sub->features.dPtr(),
                                  output.nSpatialSites, input.nFeatures,
                                  output.nFeatures, p, p, __FILE__, __LINE__);
    cudaCheckError();
  }
  multiplyAddCount += (__int128_t)output.nSpatialSites *
                      input.featuresPresent.size() *
                      output.featuresPresent.size();
  applySigmoid(output, output, fn, memStream);
  cudaCheckError();
}
void NetworkInNetworkLayer::scaleWeights(SpatiallySparseBatchInterface &input,
                                         SpatiallySparseBatchInterface &output,
                                         float &scalingUnderneath,
                                         bool topLayer) {
  assert(output.sub->features.size() > 0 && "call after forwards(...)");
  float scale = output.sub->features.meanAbs();
  std::cout << "featureScale:" << scale << std::endl;
  if (topLayer) {
    scale = 1;
  } else {
    scale = powf(scale, -0.1);
  }
  W.multiplicativeRescale(scale / scalingUnderneath);
  B.multiplicativeRescale(scale);
  MW.multiplicativeRescale(scale / scalingUnderneath);
  MB.multiplicativeRescale(scale);
  scalingUnderneath = scale;
}
void NetworkInNetworkLayer::backwards(SpatiallySparseBatch &batch,
                                      SpatiallySparseBatchInterface &input,
                                      SpatiallySparseBatchInterface &output,
                                      float learningRate, float momentum) {
  applySigmoidBackProp(output, output, fn, memStream);
  dw.resize(input.featuresPresent.size() * output.featuresPresent.size());
  db.resize(output.featuresPresent.size());
  d_rowMajorSGEMM_alphaAtB_betaC(
      cublasHandle, input.sub->features.dPtr(), output.sub->dfeatures.dPtr(),
      dw.dPtr(), input.featuresPresent.size(), output.nSpatialSites,
      output.featuresPresent.size(), 1.0, 0.0);

  multiplyAddCount += (__int128_t)output.nSpatialSites *
                      input.featuresPresent.size() *
                      output.featuresPresent.size();
  cudaCheckError();
  db.setZero(memStream);
  columnSum(output.sub->dfeatures.dPtr(), db.dPtr(), output.nSpatialSites,
            output.featuresPresent.size(), memStream);

  if (nFeaturesIn + nFeaturesOut >
      input.featuresPresent.size() + output.featuresPresent.size()) {
    if (input.backpropErrors) {
      input.sub->dfeatures.resize(input.nSpatialSites *
                                  input.featuresPresent.size());
      d_rowMajorSGEMM_alphaABt_betaC(cublasHandle, output.sub->dfeatures.dPtr(),
                                     w.dPtr(), input.sub->dfeatures.dPtr(),
                                     output.nSpatialSites,
                                     output.featuresPresent.size(),
                                     input.featuresPresent.size(), 1.0, 0.0);
      multiplyAddCount += (__int128_t)output.nSpatialSites *
                          input.featuresPresent.size() *
                          output.featuresPresent.size();
      cudaCheckError();
    }

    dGradientDescentShrunkMatrix << <input.featuresPresent.size(),
                                     KERNELBLOCKSIZE, 0, memStream.stream>>>
        (dw.dPtr(), MW.dPtr(), W.dPtr(), output.nFeatures,
         output.featuresPresent.size(), input.featuresPresent.dPtr(),
         output.featuresPresent.dPtr(), learningRate, momentum);
    cudaCheckError();

    dGradientDescentShrunkVector << <1, NTHREADS, 0, memStream.stream>>>
        (db.dPtr(), MB.dPtr(), B.dPtr(), output.nFeatures,
         output.featuresPresent.size(), output.featuresPresent.dPtr(),
         learningRate, momentum);
    cudaCheckError();
  } else {
    if (input.backpropErrors) {
      input.sub->dfeatures.resize(input.nSpatialSites *
                                  input.featuresPresent.size());
      d_rowMajorSGEMM_alphaABt_betaC(cublasHandle, output.sub->dfeatures.dPtr(),
                                     W.dPtr(), input.sub->dfeatures.dPtr(),
                                     output.nSpatialSites, nFeaturesOut,
                                     nFeaturesIn, 1.0, 0.0);
      multiplyAddCount += (__int128_t)output.nSpatialSites *
                          input.featuresPresent.size() *
                          output.featuresPresent.size();
      cudaCheckError();
    }
    dGradientDescent << <nFeaturesIn, KERNELBLOCKSIZE, 0, memStream.stream>>>
        (dw.dPtr(), MW.dPtr(), W.dPtr(), nFeaturesOut, learningRate, momentum);
    cudaCheckError();
    dGradientDescent << <1, KERNELBLOCKSIZE, 0, memStream.stream>>>
        (db.dPtr(), MB.dPtr(), B.dPtr(), nFeaturesOut, learningRate, momentum);
    cudaCheckError();
  }
}
void NetworkInNetworkLayer::loadWeightsFromStream(std::ifstream &f,
                                                  bool momentum) {
  W.copyToCPUAsync(memStream);
  MW.copyToCPUAsync(memStream);
  B.copyToCPUAsync(memStream);
  MB.copyToCPUAsync(memStream);

  f.read((char *)&W.hVector()[0], sizeof(float) * W.size());
  f.read((char *)&B.hVector()[0], sizeof(float) * B.size());
  if (momentum) {
    f.read((char *)&MW.hVector()[0], sizeof(float) * MW.size());
    f.read((char *)&MB.hVector()[0], sizeof(float) * MB.size());
  } else {
    MW.setZero();
    MB.setZero();
  }

  W.copyToGPUAsync(memStream);
  MW.copyToGPUAsync(memStream);
  B.copyToGPUAsync(memStream);
  MB.copyToGPUAsync(memStream);
};
void NetworkInNetworkLayer::putWeightsToStream(std::ofstream &f,
                                               bool momentum) {
  W.copyToCPUAsync(memStream);
  MW.copyToCPUAsync(memStream);
  B.copyToCPUAsync(memStream);
  MB.copyToCPUAsync(memStream);
  f.write((char *)&W.hVector()[0], sizeof(float) * W.size());
  f.write((char *)&B.hVector()[0], sizeof(float) * B.size());
  if (momentum) {
    f.write((char *)&MW.hVector()[0], sizeof(float) * MW.size());
    f.write((char *)&MB.hVector()[0], sizeof(float) * MB.size());
  }
  W.copyToGPUAsync(memStream);
  MW.copyToGPUAsync(memStream);
  B.copyToGPUAsync(memStream);
  MB.copyToGPUAsync(memStream);
};
int NetworkInNetworkLayer::calculateInputSpatialSize(int outputSpatialSize) {
  return outputSpatialSize;
}
