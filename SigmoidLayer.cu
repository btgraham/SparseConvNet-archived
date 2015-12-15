#include <iostream>
#include "SigmoidLayer.h"
#include "utilities.h"

////////////////////////////////////////////////////////////////////////////////
__global__ void dSigmoidReLu(float *a, float *b, int nOut) {
  int i = blockIdx.x * nOut;
  for (int j = i + threadIdx.x; j < i + nOut; j += KERNELBLOCKSIZE) {
    b[j] = (a[j] > 0) ? a[j] : 0;
  }
}
void sigmoidReLU(float *a, float *b, int count, int nOut,
                 cudaMemStream &memStream) {
  int processed = 0;
  while (processed < count) {
    int batch = min(32768 / 4, count - processed);
    dSigmoidReLu << <batch, KERNELBLOCKSIZE, 0, memStream.stream>>>
        (a + processed * nOut, b + processed * nOut, nOut);
    processed += batch;
  }
  cudaCheckError();
}
__global__ void dSigmoidBackpropReLu(float *a, float *b, float *da, float *db,
                                     int nOut) {
  int i = blockIdx.x * nOut;
  for (int j = i + threadIdx.x; j < i + nOut; j += KERNELBLOCKSIZE) {
    da[j] = (a[j] > 0) ? db[j] : 0;
  }
}
void sigmoidBackpropReLU(float *a, float *b, float *da, float *db, int count,
                         int nOut, cudaMemStream &memStream) {
  int processed = 0;
  while (processed < count) {
    int batch = min(32768 / 4, count - processed);
    dSigmoidBackpropReLu << <batch, KERNELBLOCKSIZE, 0, memStream.stream>>>
        (a + processed * nOut, b + processed * nOut, da + processed * nOut,
         db + processed * nOut, nOut);
    processed += batch;
  }
  cudaCheckError();
}
////////////////////////////////////////////////////////////////////////////////
__global__ void dSigmoidLogistic(float *a, float *b, int nOut) {
  int i = blockIdx.x * nOut;
  for (int j = i + threadIdx.x; j < i + nOut; j += KERNELBLOCKSIZE) {
    b[j] = 1.0f / (1.0f + exp(-a[j]));
  }
}
void sigmoidLogistic(float *a, float *b, int count, int nOut,
                     cudaMemStream &memStream) {
  int processed = 0;
  while (processed < count) {
    int batch = min(32768 / 4, count - processed);
    dSigmoidLogistic << <batch, KERNELBLOCKSIZE, 0, memStream.stream>>>
        (a + processed * nOut, b + processed * nOut, nOut);
    processed += batch;
  }
  cudaCheckError();
}
__global__ void dSigmoidBackpropLogistic(float *a, float *b, float *da,
                                         float *db, int nOut) {
  int i = blockIdx.x * nOut;
  for (int j = i + threadIdx.x; j < i + nOut; j += KERNELBLOCKSIZE) {
    da[j] = db[j] * b[j] * (1 - b[j]);
  }
}
void sigmoidBackpropLogistic(float *a, float *b, float *da, float *db,
                             int count, int nOut, cudaMemStream &memStream) {
  int processed = 0;
  while (processed < count) {
    int batch = min(32768 / 4, count - processed);
    dSigmoidBackpropLogistic << <batch, KERNELBLOCKSIZE, 0, memStream.stream>>>
        (a + processed * nOut, b + processed * nOut, da + processed * nOut,
         db + processed * nOut, nOut);
    processed += batch;
  }
  cudaCheckError();
}
////////////////////////////////////////////////////////////////////////////////
__global__ void dSigmoidTanh(float *a, float *b, int nOut) {
  int i = blockIdx.x * nOut;
  for (int j = i + threadIdx.x; j < i + nOut; j += KERNELBLOCKSIZE) {
    b[j] = tanhf(a[j]);
  }
}
void sigmoidTanh(float *a, float *b, int count, int nOut,
                 cudaMemStream &memStream) {
  int processed = 0;
  while (processed < count) {
    int batch = min(32768 / 4, count - processed);
    dSigmoidTanh << <batch, KERNELBLOCKSIZE, 0, memStream.stream>>>
        (a + processed * nOut, b + processed * nOut, nOut);
    processed += batch;
  }
  cudaCheckError();
}

__global__ void dSigmoidBackpropTanh(float *a, float *b, float *da, float *db,
                                     int nOut) {
  int i = blockIdx.x * nOut;
  for (int j = i + threadIdx.x; j < i + nOut; j += KERNELBLOCKSIZE) {
    da[j] = db[j] * (1 + b[j]) * (1 - b[j]);
  }
}
void sigmoidBackpropTanh(float *a, float *b, float *da, float *db, int count,
                         int nOut, cudaMemStream &memStream) {
  int processed = 0;
  while (processed < count) {
    int batch = min(32768 / 4, count - processed);
    dSigmoidBackpropTanh << <batch, KERNELBLOCKSIZE, 0, memStream.stream>>>
        (a + processed * nOut, b + processed * nOut, da + processed * nOut,
         db + processed * nOut, nOut);
    processed += batch;
  }
  cudaCheckError();
}
////////////////////////////////////////////////////////////////////////////////
__global__ void dSigmoidLeakyReLu(float *a, float *b, int nOut, float alpha) {
  int i = blockIdx.x * nOut;
  for (int j = i + threadIdx.x; j < i + nOut; j += KERNELBLOCKSIZE) {
    b[j] = (a[j] > 0) ? a[j] : (a[j] * alpha);
  }
}
void sigmoidLeakyReLU(float *a, float *b, int count, int nOut,
                      float alpha, // 0.01 or 0.3 say
                      cudaMemStream &memStream) {
  int processed = 0;
  while (processed < count) {
    int batch = min(32768, count - processed);
    dSigmoidLeakyReLu << <batch, KERNELBLOCKSIZE, 0, memStream.stream>>>
        (a + processed * nOut, b + processed * nOut, nOut, alpha);
    processed += batch;
  }
  cudaCheckError();
}
__global__ void dSigmoidBackpropLeakyReLu(float *a, float *b, float *da,
                                          float *db, int nOut, float alpha) {
  int i = blockIdx.x * nOut;
  for (int j = i + threadIdx.x; j < i + nOut; j += KERNELBLOCKSIZE) {
    da[j] = (a[j] > 0) ? db[j] : (db[j] * alpha);
    __syncthreads();
  }
}
void sigmoidBackpropLeakyReLU(float *a, float *b, float *da, float *db,
                              int count, int nOut, float alpha,
                              cudaMemStream &memStream) {
  int processed = 0;
  while (processed < count) {
    int batch = min(32768, count - processed);
    dSigmoidBackpropLeakyReLu << <batch, KERNELBLOCKSIZE, 0, memStream.stream>>>
        (a + processed * nOut, b + processed * nOut, da + processed * nOut,
         db + processed * nOut, nOut, alpha);
    processed += batch;
  }
  cudaCheckError();
}
////////////////////////////////////////////////////////////////////////////////
// SOFTMAX should only be used in the top layer;
// derivative contained in calculation of initial d_delta.
__global__ void dSigmoidSoftmax(float *a, float *b, int count, int nOut) {
  for (int i = threadIdx.x; i < count; i += NTHREADS) {
    float acc = 0.0f;
    float mx = -10000.0f;
    for (int k = 0; k < nOut; k++)
      if (a[i * nOut + k] > mx)
        mx = a[i * nOut + k];
    for (int k = 0; k < nOut; k++) {
      b[i * nOut + k] =
          expf((a[i * nOut + k] -
                mx)); // Subtract row max value for numerical stability.
      acc += b[i * nOut + k];
    }
    for (int k = 0; k < nOut; k++) {
      b[i * nOut + k] /= acc;
    }
  }
}
__global__ void dSigmoidBackpropSoftmax(float *a, float *b, float *da,
                                        float *db, int count, int nOut) {
  for (int i = 0; i < count; i++) {
    for (int k = threadIdx.x; k < nOut; k += NTHREADS) {
      da[i * nOut + k] = db[i * nOut + k];
    }
  }
}
////////////////////////////////////////////////////////////////////////////////
void applySigmoid(SpatiallySparseBatchInterface &input,
                  SpatiallySparseBatchInterface &output, ActivationFunction fn,
                  cudaMemStream &memStream) {
  switch (fn) {
  case SIGMOID:
    sigmoidLogistic(input.sub->features.dPtr(), output.sub->features.dPtr(),
                    output.nSpatialSites, output.featuresPresent.size(),
                    memStream);
    break;
  case TANH:
    sigmoidTanh(input.sub->features.dPtr(), output.sub->features.dPtr(),
                output.nSpatialSites, output.featuresPresent.size(), memStream);
    break;
  case RELU:
    sigmoidReLU(input.sub->features.dPtr(), output.sub->features.dPtr(),
                output.nSpatialSites, output.featuresPresent.size(), memStream);
    break;
  case LEAKYRELU:
    sigmoidLeakyReLU(input.sub->features.dPtr(), output.sub->features.dPtr(),
                     output.nSpatialSites, output.featuresPresent.size(), 0.01,
                     memStream);
    break;
  case VLEAKYRELU:
    sigmoidLeakyReLU(input.sub->features.dPtr(), output.sub->features.dPtr(),
                     output.nSpatialSites, output.featuresPresent.size(), 0.333,
                     memStream);
    break;
  case SOFTMAX:
    dSigmoidSoftmax << <1, NTHREADS, 0, memStream.stream>>>
        (input.sub->features.dPtr(), output.sub->features.dPtr(),
         output.nSpatialSites, output.featuresPresent.size());
    break;
  case NOSIGMOID:
    break;
  }
}

void applySigmoidBackProp(SpatiallySparseBatchInterface &input,
                          SpatiallySparseBatchInterface &output,
                          ActivationFunction fn, cudaMemStream &memStream) {
  switch (fn) {
  case SIGMOID:
    sigmoidBackpropLogistic(
        input.sub->features.dPtr(), output.sub->features.dPtr(),
        input.sub->dfeatures.dPtr(), output.sub->dfeatures.dPtr(),
        output.nSpatialSites, output.featuresPresent.size(), memStream);
    break;
  case TANH:
    sigmoidBackpropTanh(input.sub->features.dPtr(), output.sub->features.dPtr(),
                        input.sub->dfeatures.dPtr(),
                        output.sub->dfeatures.dPtr(), output.nSpatialSites,
                        output.featuresPresent.size(), memStream);
    break;
  case RELU:
    sigmoidBackpropReLU(input.sub->features.dPtr(), output.sub->features.dPtr(),
                        input.sub->dfeatures.dPtr(),
                        output.sub->dfeatures.dPtr(), output.nSpatialSites,
                        output.featuresPresent.size(), memStream);
    break;
  case LEAKYRELU:
    sigmoidBackpropLeakyReLU(
        input.sub->features.dPtr(), output.sub->features.dPtr(),
        input.sub->dfeatures.dPtr(), output.sub->dfeatures.dPtr(),
        output.nSpatialSites, output.featuresPresent.size(), 0.01, memStream);
    break;
  case VLEAKYRELU:
    sigmoidBackpropLeakyReLU(
        input.sub->features.dPtr(), output.sub->features.dPtr(),
        input.sub->dfeatures.dPtr(), output.sub->dfeatures.dPtr(),
        output.nSpatialSites, output.featuresPresent.size(), 0.333, memStream);
    break;
  case SOFTMAX:
    dSigmoidBackpropSoftmax << <1, NTHREADS, 0, memStream.stream>>>
        (input.sub->features.dPtr(), output.sub->features.dPtr(),
         input.sub->dfeatures.dPtr(), output.sub->dfeatures.dPtr(),
         output.nSpatialSites, output.featuresPresent.size());
    break;
  case NOSIGMOID:
    break;
  }
}

SigmoidLayer::SigmoidLayer(cudaMemStream &memStream, ActivationFunction fn)
    : SpatiallySparseLayer(memStream), fn(fn) {
  std::cout << sigmoidNames[fn] << std::endl;
};
void SigmoidLayer::preprocess(SpatiallySparseBatch &batch,
                              SpatiallySparseBatchInterface &input,
                              SpatiallySparseBatchInterface &output) {
  output.nFeatures = input.nFeatures;
  output.featuresPresent.hVector() = input.featuresPresent.hVector();
  output.spatialSize = input.spatialSize;
  output.nSpatialSites = input.nSpatialSites;
  output.grids = input.grids;
  output.backpropErrors = input.backpropErrors;
}
void SigmoidLayer::forwards(SpatiallySparseBatch &batch,
                            SpatiallySparseBatchInterface &input,
                            SpatiallySparseBatchInterface &output) {
  output.sub->features.resize(output.nSpatialSites *
                              output.featuresPresent.size());
  applySigmoid(input, output, fn, memStream);
}
void SigmoidLayer::backwards(SpatiallySparseBatch &batch,
                             SpatiallySparseBatchInterface &input,
                             SpatiallySparseBatchInterface &output,
                             float learningRate, float momentum) {
  if (input.backpropErrors) {
    input.sub->dfeatures.resize(input.nSpatialSites *
                                input.featuresPresent.size());
    applySigmoidBackProp(input, output, fn, memStream);
  }
}
int SigmoidLayer::calculateInputSpatialSize(int outputSpatialSize) {
  return outputSpatialSize;
}
