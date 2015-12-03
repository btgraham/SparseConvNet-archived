// Ben Graham, University of Warwick, 2015 b.graham@warwick.ac.uk
// SparseConvNet is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
//(at your option) any later version.

// SparseConvNet is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with SparseConvNet.  If not, see <http://www.gnu.org/licenses/>.

#pragma once
#include "SpatiallySparseBatch.h"
#include "SpatiallySparseLayer.h"
#include "SpatiallySparseDataset.h"
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#define N_MAX_BATCH_PRODUCER_THREADS 20

class SparseConvNetCUDA {
public:
  std::vector<SpatiallySparseLayer *> layers;
  std::vector<float> inputNormalizingConstants;
  int dimension;
  int nClasses;
  int nTop;
  int inputSpatialSize;
  int nInputFeatures;
  int nOutputFeatures;
  int deviceID;
  int nBatchProducerThreads;
  cudaMemStream memStream;
  cudaMemStream batchMemStreams[N_MAX_BATCH_PRODUCER_THREADS];
  std::mutex batchLock[N_MAX_BATCH_PRODUCER_THREADS];
  std::vector<SpatiallySparseBatch> batchPool;
  std::vector<SpatiallySparseBatchSubInterface *> initialSubInterfaces;
  std::vector<SpatiallySparseBatchSubInterface *> sharedSubInterfaces;
  cublasHandle_t cublasHandle;

  SparseConvNetCUDA(int dimension, int nInputFeatures, int nClasses,
                    int pciBusID = -1, int nTop = 1,
                    int nBatchProducerThreads = 10);
  ~SparseConvNetCUDA();
  void processBatch(SpatiallySparseBatch &batch, float learningRate,
                    float momentum, std::ofstream &f, std::ofstream &g);
  void processIndexLearnerBatch(SpatiallySparseBatch &batch, float learningRate,
                                float momentum, std::ofstream &f);

  void addLearntLayer(int nFeatures, ActivationFunction activationFn = RELU,
                      float dropout = 0.0f, float alpha = 1.0f);
  void addNetworkInNetworkLayer(int nFeatures,
                                ActivationFunction activationFn = RELU,
                                float dropout = 0.0f);
  void addConvolutionalLayer(int nFeatures, int filterSize, int filterStride,
                             ActivationFunction activationFn = RELU,
                             float dropout = 0.0f, int minActiveInputs = 1,
                             float poolingToFollow = 1.0f);
  void addLeNetLayerMP(int nFeatures, int filterSize, int filterStride,
                       int poolSize, int poolStride,
                       ActivationFunction activationFn = RELU,
                       float dropout = 0.0f, int minActiveInputs = 1);
  void addLeNetLayerROFMP(int nFeatures, int filterSize, int filterStride,
                          int poolSize, float fmpShrink,
                          ActivationFunction activationFn = RELU,
                          float dropout = 0.0f, int minActiveInputs = 1);
  void addLeNetLayerPOFMP(int nFeatures, int filterSize, int filterStride,
                          int poolSize, float fmpShrink,
                          ActivationFunction activationFn = RELU,
                          float dropout = 0.0f, int minActiveInputs = 1);
  void addLeNetLayerRDFMP(int nFeatures, int filterSize, int filterStride,
                          int poolSize, float fmpShrink,
                          ActivationFunction activationFn = RELU,
                          float dropout = 0.0f, int minActiveInputs = 1);
  void addLeNetLayerPDFMP(int nFeatures, int filterSize, int filterStride,
                          int poolSize, float fmpShrink,
                          ActivationFunction activationFn = RELU,
                          float dropout = 0.0f, int minActiveInputs = 1);
  void addTriangularConvolutionalLayer(int nFeatures, int filterSize,
                                       int filterStride,
                                       ActivationFunction activationFn = RELU,
                                       float dropout = 0.0f,
                                       int minActiveInputs = 1,
                                       float poolingToFollow = 1.0f);
  void addTriangularLeNetLayerMP(int nFeatures, int filterSize,
                                 int filterStride, int poolSize, int poolStride,
                                 ActivationFunction activationFn = RELU,
                                 float dropout = 0.0f, int minActiveInputs = 1);
  void addSoftmaxLayer();
  void addTerminalPoolingLayer(int poolSize, int S);
  void addIndexLearnerLayer();
  float processDataset(SpatiallySparseDataset &dataset, int batchSize = 100,
                       float learningRate = 0, float momentum = 0.99);
  void processDatasetRepeatTest(SpatiallySparseDataset &dataset,
                                int batchSize = 100, int nReps = 12,
                                std::string predictionsFilename = "",
                                std::string confusionMatrixFilename = "");
  float processIndexLearnerDataset(SpatiallySparseDataset &dataset,
                                   int batchSize = 100,
                                   float learningRate = 0.0,
                                   float momentum = 0.99);
  void processBatchDumpTopLevelFeaturess(SpatiallySparseBatch &batch,
                                         std::ofstream &f);
  void processDatasetDumpTopLevelFeatures(SpatiallySparseDataset &dataset,
                                          int batchSize, int reps = 1);
  void loadWeights(std::string baseName, int epoch, bool momentum,
                   int firstNlayers = 1000000);
  void saveWeights(std::string baseName, int epoch, bool momentum);
  void calculateInputRegularizingConstants(SpatiallySparseDataset dataset);
};
