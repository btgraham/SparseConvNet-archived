//Ben Graham, University of Warwick, 2015 b.graham@warwick.ac.uk
//SparseConvNet is free software: you can redistribute it and/or modify
//it under the terms of the GNU General Public License as published by
//the Free Software Foundation, either version 3 of the License, or
//(at your option) any later version.

//SparseConvNet is distributed in the hope that it will be useful,
//but WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//GNU General Public License for more details.

//You should have received a copy of the GNU General Public License
//along with SparseConvNet.  If not, see <http://www.gnu.org/licenses/>.

#pragma once
#include "SpatiallySparseBatch.h"
#include "SpatiallySparseLayer.h"
#include "SpatiallySparseDataset.h"
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

class SparseConvNetCUDA {
public:
  std::vector<SpatiallySparseLayer*> layers;
  int dimension;
  int nClasses;
  int nTop;
  int inputSpatialSize;
  int nInputFeatures;
  int nOutputFeatures;
  int deviceID;
  std::vector<float> inputNormalizingConstants;
  SparseConvNetCUDA (int dimension,
                     int nInputFeatures,
                     int nClasses,
                     int pciBusID=-1,
                     int nTop=1);
  void processBatch(SpatiallySparseBatch& batch, float learningRate, std::ofstream& f, std::ofstream& g);
  void processIndexLearnerBatch(SpatiallySparseBatch& batch, float learningRate, std::ofstream& f);



  void addLearntLayer(int nFeatures,
                      ActivationFunction activationFn=RELU,
                      float dropout=0.0f,
                      float alpha=1.0f);
  void addNetworkInNetworkLayer(int nFeatures,
                                ActivationFunction activationFn=RELU,
                                float dropout=0.0f);
  void addConvolutionalLayer(int nFeatures,
                             int filterSize,
                             int filterStride,
                             ActivationFunction activationFn=RELU,
                             float dropout=0.0f,
                             float poolingToFollow=1.0f);
  void addLeNetLayerMP(int nFeatures, int filterSize, int filterStride, int poolSize, int poolStride, ActivationFunction activationFn=RELU, float dropout=0.0f);
  void addLeNetLayerROFMP(int nFeatures, int filterSize, int filterStride, int poolSize, float fmpShrink, ActivationFunction activationFn=RELU, float dropout=0.0f);
  void addLeNetLayerPOFMP(int nFeatures, int filterSize, int filterStride, int poolSize, float fmpShrink, ActivationFunction activationFn=RELU, float dropout=0.0f);
  void addTriangularConvolutionalLayer(int nFeatures,
                                       int filterSize,
                                       int filterStride,
                                       ActivationFunction activationFn=RELU,
                                       float dropout=0.0f,
                                       float poolingToFollow=1.0f);
  void addTriangularLeNetLayerMP(int nFeatures, int filterSize, int filterStride, int poolSize, int poolStride, ActivationFunction activationFn=RELU, float dropout=0.0f);
  void addSoftmaxLayer();
  void addTerminalPoolingLayer(int poolSize, int S);
  void addIndexLearnerLayer();
  float processDataset(SpatiallySparseDataset &dataset, int batchSize=100, float learningRate=0);
  void processDatasetRepeatTest(SpatiallySparseDataset &dataset, int batchSize=100, int nReps=12, std::string predictionsFilename="",std::string header="");
  float processIndexLearnerDataset(SpatiallySparseDataset &dataset, int batchSize=100, float learningRate=0.0);
  void processBatchDumpTopLevelFeaturess(SpatiallySparseBatch& batch, std::ofstream& f);
  void processDatasetDumpTopLevelFeatures(SpatiallySparseDataset &dataset, int batchSize, int reps=1);
  void loadWeights(std::string baseName, int epoch, int firstNlayers=1000000);
  void saveWeights(std::string baseName, int epoch);
  void calculateInputRegularizingConstants(SpatiallySparseDataset dataset);
};
