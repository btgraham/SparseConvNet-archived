// Ben Graham, University of Warwick, 2015, b.graham@warwick.ac.uk
// SparseConvNet is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
//(at your option) any later version.

// SparseConvNet is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

#pragma once
#include "SpatiallySparseDataset.h"
#include <memory>
#include <string>
#include <fstream>

class SparseConvNetCUDA;

class SparseConvNet {
private:
public:
  std::unique_ptr<SparseConvNetCUDA> cnn;

  SparseConvNet(int dimension, int nInputFeatures, int nClasses,
                int pciBusID = -1, int nTop = 1);
  ~SparseConvNet();
  void addLeNetLayerMP(int nFeatures, int filterSize, int filterStride,
                       int poolSize, int poolStride,
                       ActivationFunction activationFn = RELU,
                       float dropout = 0.0f, int minActiveInputs = 1);
  void addLeNetLayerPOFMP(int nFeatures, int filterSize, int filterStride,
                          int poolSize, float fmpShrink,
                          ActivationFunction activationFn = RELU,
                          float dropout = 0.0f, int minActiveInputs = 1);
  void addLeNetLayerROFMP(int nFeatures, int filterSize, int filterStride,
                          int poolSize, float fmpShrink,
                          ActivationFunction activationFn = RELU,
                          float dropout = 0.0f, int minActiveInputs = 1);
  void addLeNetLayerPDFMP(int nFeatures, int filterSize, int filterStride,
                          int poolSize, float fmpShrink,
                          ActivationFunction activationFn = RELU,
                          float dropout = 0.0f, int minActiveInputs = 1);
  void addLeNetLayerRDFMP(int nFeatures, int filterSize, int filterStride,
                          int poolSize, float fmpShrink,
                          ActivationFunction activationFn = RELU,
                          float dropout = 0.0f, int minActiveInputs = 1);
  void addTerminalPoolingLayer(int poolSize);
  void addSoftmaxLayer();
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
  void processDatasetDumpTopLevelFeatures(SpatiallySparseDataset &dataset,
                                          int batchSize, int reps = 1);
  void loadWeights(std::string baseName, int epoch, bool momentum = false,
                   int firstNlayers = 1000000);
  void saveWeights(std::string baseName, int epoch, bool momentum = false);
  void calculateInputRegularizingConstants(SpatiallySparseDataset dataset);
};

class SparseConvTriangLeNet {
private:
  std::auto_ptr<SparseConvNetCUDA> cnn;

public:
  SparseConvTriangLeNet(int dimension, int nInputFeatures, int nClasses,
                        int pciBusID = -1, int nTop = 1);
  ~SparseConvTriangLeNet();
  void addLeNetLayerMP(int nFeatures, int filterSize, int filterStride,
                       int poolSize, int poolStride,
                       ActivationFunction activationFn = RELU,
                       float dropout = 0.0f, int minActiveInputs = 1);
  void addTerminalPoolingLayer(int poolSize);
  void addSoftmaxLayer();
  void addIndexLearnerLayer();
  float processDataset(SpatiallySparseDataset &dataset, int batchSize = 100,
                       float learningRate = 0, float momentum = 0.99);
  void processDatasetRepeatTest(SpatiallySparseDataset &dataset,
                                int batchSize = 100, int nReps = 12,
                                std::string predictionsFilename = "",
                                std::string confusionMatrixFilename = "");
  float processIndexLearnerDataset(SpatiallySparseDataset &dataset,
                                   int batchSize = 100, float learningRate = 0,
                                   float momentum = 0.99);
  void processDatasetDumpTopLevelFeatures(SpatiallySparseDataset &dataset,
                                          int batchSize, int reps = 1);
  void loadWeights(std::string baseName, int epoch, bool momentum = false,
                   int firstNlayers = 1000000);
  void saveWeights(std::string baseName, int epoch, bool momentum = false);
  void calculateInputRegularizingConstants(SpatiallySparseDataset dataset);
};
