#include "NetworkInNetworkPReLULayer.h"
#include "utilities.h"
#include <iostream>
#include <cassert>

const float initialAlpha = 0.25;

__global__ void dPReLU(float *features, float *prelu, int nRows, int nColumns) {
  int c = blockIdx.x * KERNELBLOCKSIZE + threadIdx.x;
  float alpha = prelu[c];
  for (int i = blockIdx.y; i < nRows; i += 1024) {
    float f = features[i * nColumns + c];
    features[i * nColumns + c] = (f > 0) ? f : (f * alpha);
  }
}
void PreLU(float *matrix, float *prelu, int nRows, int nColumns,
           cudaMemStream &memStream) {
  assert(nColumns % KERNELBLOCKSIZE == 0);
  dPReLU << <dim3(nColumns / KERNELBLOCKSIZE, std::min(1024, nRows)),
             KERNELBLOCKSIZE, 0, memStream.stream>>>
      (matrix, prelu, nRows, nColumns);
  cudaCheckError();
}

__global__ void dPReLUBackprop(float *features, float *dfeatures, float *prelu,
                               float *dprelu, int nRows, int nColumns) {
  int c = blockIdx.x * KERNELBLOCKSIZE + threadIdx.x;
  float alpha = prelu[c];
  float t = 0;
  for (int j = blockIdx.y; j < nRows; j += KERNELBLOCKSIZE) {
    float b = features[j * nColumns + c];
    float d = dfeatures[j * nColumns + c];
    t += d * ((b > 0) ? (0) : (b / alpha));

    dfeatures[j * nColumns + c] = ((b > 0) ? (d) : (d * alpha));
  }
  atomicAdd(&dprelu[c], t);
}

void PReLUBackprop(float *features, float *dfeatures, float *prelu,
                   float *dprelu, int nRows, int nColumns,
                   cudaMemStream &memStream) {
  dPReLUBackprop << <dim3(nColumns / KERNELBLOCKSIZE, KERNELBLOCKSIZE),
                     KERNELBLOCKSIZE, 0, memStream.stream>>>
      (features, dfeatures, prelu, dprelu, nRows, nColumns);
  cudaCheckError();
}

__global__ void dGradientDescentShrunkVectorKeepPositive(
    float *d_delta, float *d_momentum, float *d_weights, int nOut,
    int nOutDropout, int *outFeaturesPresent, float learningRate,
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
    w = (w > 0.001) ? w : 0.001;
    d_weights[ii] = w;
    d_momentum[ii] = m;
  }
}

__global__ void dGradientDescentKeepPositive(float *d_delta, float *d_momentum,
                                             float *d_weights, int nOut,
                                             float learningRate,
                                             float momentum) {
  int i = blockIdx.x * nOut;
  for (int j = i + threadIdx.x; j < i + nOut; j += KERNELBLOCKSIZE) {
    float w = d_weights[j];
    float m = d_momentum[j];
    float d = d_delta[j];
    w -= m * momentum;
    m = momentum * m - learningRate * (1 - momentum) * d;
    w += m * (1 + momentum);
    w = (w > 0.001) ? w : 0.001;
    d_weights[j] = w;
    d_momentum[j] = m;
  }
}

NetworkInNetworkPReLULayer::NetworkInNetworkPReLULayer(
    cudaMemStream &memStream, cublasHandle_t &cublasHandle, int nFeaturesIn,
    int nFeaturesOut, float dropout,
    float alpha // used to determine intialization weights only
    )
    : SpatiallySparseLayer(memStream), cublasHandle(cublasHandle),
      nFeaturesIn(nFeaturesIn), nFeaturesOut(nFeaturesOut), dropout(dropout),
      W(true, nFeaturesIn * nFeaturesOut), MW(true, nFeaturesIn * nFeaturesOut),
      B(true, nFeaturesOut), MB(true, nFeaturesOut), PReLU(true, nFeaturesOut),
      MPReLU(true, nFeaturesOut) {
  float scale = pow(6.0f / (nFeaturesIn + nFeaturesOut * alpha), 0.5f);
  W.copyToCPUAsync(memStream);
  W.setUniform(-scale, scale);
  W.copyToGPUAsync(memStream);
  PReLU.copyToCPUAsync(memStream);
  PReLU.setConstant(initialAlpha);
  PReLU.copyToGPUAsync(memStream);
  MW.setZero();
  B.setZero();
  MB.setZero();
  MPReLU.setZero();
  std::cout << "Learn " << nFeaturesIn << "->" << nFeaturesOut
            << " dropout=" << dropout << " " << sigmoidNames[PRELU]
            << std::endl;
}
void NetworkInNetworkPReLULayer::preprocess(
    SpatiallySparseBatch &batch, SpatiallySparseBatchInterface &input,
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
void NetworkInNetworkPReLULayer::forwards(
    SpatiallySparseBatch &batch, SpatiallySparseBatchInterface &input,
    SpatiallySparseBatchInterface &output) {
  // std::cerr << output.nFeatures << " " << PReLU.meanAbs() << "\n";
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
    prelu.resize(output.featuresPresent.size());
    dShrinkVectorForDropout << <1, NTHREADS, 0, memStream.stream>>>
        (PReLU.dPtr(), prelu.dPtr(), output.featuresPresent.dPtr(),
         output.nFeatures, output.featuresPresent.size());
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
    PreLU(output.sub->features.dPtr(), prelu.dPtr(), output.nSpatialSites,
          output.featuresPresent.size(), memStream);
  } else {
    float p = 1.0f - (batch.type != RESCALEBATCH ? dropout : 0);
    replicateArray(B.dPtr(), output.sub->features.dPtr(), output.nSpatialSites,
                   output.featuresPresent.size(), memStream);
    d_rowMajorSGEMM_alphaAB_betaC(cublasHandle, input.sub->features.dPtr(),
                                  W.dPtr(), output.sub->features.dPtr(),
                                  output.nSpatialSites, input.nFeatures,
                                  output.nFeatures, p, p, __FILE__, __LINE__);
    cudaCheckError();
    PreLU(output.sub->features.dPtr(), PReLU.dPtr(), output.nSpatialSites,
          output.nFeatures, memStream);
  }
  multiplyAddCount += (__int128_t)output.nSpatialSites *
                      input.featuresPresent.size() *
                      output.featuresPresent.size();
}
void NetworkInNetworkPReLULayer::scaleWeights(
    SpatiallySparseBatchInterface &input, SpatiallySparseBatchInterface &output,
    float &scalingUnderneath, bool topLayer) {
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
void NetworkInNetworkPReLULayer::backwards(
    SpatiallySparseBatch &batch, SpatiallySparseBatchInterface &input,
    SpatiallySparseBatchInterface &output, float learningRate, float momentum) {
  dw.resize(input.featuresPresent.size() * output.featuresPresent.size());
  db.resize(output.featuresPresent.size());
  dprelu.resize(output.featuresPresent.size());
  dprelu.setZero(memStream);
  if (output.featuresPresent.size() < output.featuresPresent.size()) {
    PReLUBackprop(output.sub->features.dPtr(), output.sub->dfeatures.dPtr(),
                  prelu.dPtr(), dprelu.dPtr(), output.nSpatialSites,
                  output.featuresPresent.size(), memStream);
    dGradientDescentShrunkVectorKeepPositive
            << <1, NTHREADS, 0, memStream.stream>>>
        (dprelu.dPtr(), MPReLU.dPtr(), PReLU.dPtr(), output.nFeatures,
         output.featuresPresent.size(), output.featuresPresent.dPtr(),
         learningRate, momentum);
  } else {
    PReLUBackprop(output.sub->features.dPtr(), output.sub->dfeatures.dPtr(),
                  PReLU.dPtr(), dprelu.dPtr(), output.nSpatialSites,
                  output.featuresPresent.size(), memStream);
    dGradientDescentKeepPositive << <1, KERNELBLOCKSIZE, 0, memStream.stream>>>
        (dprelu.dPtr(), MPReLU.dPtr(), PReLU.dPtr(), nFeaturesOut, learningRate,
         momentum);
  }
  cudaCheckError();

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
void NetworkInNetworkPReLULayer::loadWeightsFromStream(std::ifstream &f,
                                                       bool momentum) {
  W.copyToCPUAsync(memStream);
  MW.copyToCPUAsync(memStream);
  B.copyToCPUAsync(memStream);
  MB.copyToCPUAsync(memStream);
  PReLU.copyToCPUAsync(memStream);
  MPReLU.copyToCPUAsync(memStream);

  f.read((char *)&W.hVector()[0], sizeof(float) * W.size());
  f.read((char *)&B.hVector()[0], sizeof(float) * B.size());
  f.read((char *)&PReLU.hVector()[0], sizeof(float) * PReLU.size());
  if (momentum) {
    f.read((char *)&MW.hVector()[0], sizeof(float) * MW.size());
    f.read((char *)&MB.hVector()[0], sizeof(float) * MB.size());
    f.read((char *)&MPReLU.hVector()[0], sizeof(float) * MPReLU.size());
  } else {
    MW.setZero();
    MB.setZero();
    MPReLU.setZero();
  }

  W.copyToGPUAsync(memStream);
  MW.copyToGPUAsync(memStream);
  B.copyToGPUAsync(memStream);
  MB.copyToGPUAsync(memStream);
  PReLU.copyToGPUAsync(memStream);
  MPReLU.copyToGPUAsync(memStream);
}
void NetworkInNetworkPReLULayer::putWeightsToStream(std::ofstream &f,
                                                    bool momentum) {
  W.copyToCPUAsync(memStream);
  MW.copyToCPUAsync(memStream);
  B.copyToCPUAsync(memStream);
  MB.copyToCPUAsync(memStream);
  PReLU.copyToCPUAsync(memStream);
  MPReLU.copyToCPUAsync(memStream);
  f.write((char *)&W.hVector()[0], sizeof(float) * W.size());
  f.write((char *)&B.hVector()[0], sizeof(float) * B.size());
  f.write((char *)&PReLU.hVector()[0], sizeof(float) * PReLU.size());
  if (momentum) {
    f.write((char *)&MW.hVector()[0], sizeof(float) * MW.size());
    f.write((char *)&MB.hVector()[0], sizeof(float) * MB.size());
    f.write((char *)&MPReLU.hVector()[0], sizeof(float) * MPReLU.size());
  }
  W.copyToGPUAsync(memStream);
  MW.copyToGPUAsync(memStream);
  B.copyToGPUAsync(memStream);
  MB.copyToGPUAsync(memStream);
  PReLU.copyToGPUAsync(memStream);
  MPReLU.copyToGPUAsync(memStream);
};
int NetworkInNetworkPReLULayer::calculateInputSpatialSize(
    int outputSpatialSize) {
  return outputSpatialSize;
}
