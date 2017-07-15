#include "SparseConvNet.h"
#include "SparseConvNetCUDA.h"
#include "utilities.h"

SparseConvNet::SparseConvNet(int dimension, int nInputFeatures, int nClasses,
                             int pciBusID, int nTop)
    : cnn(new SparseConvNetCUDA(dimension, nInputFeatures, nClasses, pciBusID,
                                nTop)) {}

SparseConvNet::~SparseConvNet() {}

void SparseConvNet::addLeNetLayerMP(int nFeatures, int filterSize,
                                    int filterStride, int poolSize,
                                    int poolStride,
                                    ActivationFunction activationFn,
                                    float dropout, int minActiveInputs) {
  cnn->addLeNetLayerMP(nFeatures, filterSize, filterStride, poolSize,
                       poolStride, activationFn, dropout, minActiveInputs);
}

void SparseConvNet::addLeNetLayerPOFMP(int nFeatures, int filterSize,
                                       int filterStride, int poolSize,
                                       float fmpShrink,
                                       ActivationFunction activationFn,
                                       float dropout, int minActiveInputs) {
  cnn->addLeNetLayerPOFMP(nFeatures, filterSize, filterStride, poolSize,
                          fmpShrink, activationFn, dropout, minActiveInputs);
}

void SparseConvNet::addLeNetLayerROFMP(int nFeatures, int filterSize,
                                       int filterStride, int poolSize,
                                       float fmpShrink,
                                       ActivationFunction activationFn,
                                       float dropout, int minActiveInputs) {
  cnn->addLeNetLayerROFMP(nFeatures, filterSize, filterStride, poolSize,
                          fmpShrink, activationFn, dropout, minActiveInputs);
}
void SparseConvNet::addLeNetLayerPDFMP(int nFeatures, int filterSize,
                                       int filterStride, int poolSize,
                                       float fmpShrink,
                                       ActivationFunction activationFn,
                                       float dropout, int minActiveInputs) {
  cnn->addLeNetLayerPDFMP(nFeatures, filterSize, filterStride, poolSize,
                          fmpShrink, activationFn, dropout, minActiveInputs);
}

void SparseConvNet::addLeNetLayerRDFMP(int nFeatures, int filterSize,
                                       int filterStride, int poolSize,
                                       float fmpShrink,
                                       ActivationFunction activationFn,
                                       float dropout, int minActiveInputs) {
  cnn->addLeNetLayerRDFMP(nFeatures, filterSize, filterStride, poolSize,
                          fmpShrink, activationFn, dropout, minActiveInputs);
}

void SparseConvNet::addTerminalPoolingLayer(int poolSize) {
  cnn->addTerminalPoolingLayer(poolSize, ipow(poolSize, cnn->dimension));
}

void SparseConvNet::addSoftmaxLayer() { cnn->addSoftmaxLayer(); }

void SparseConvNet::addIndexLearnerLayer() { cnn->addIndexLearnerLayer(); }

float SparseConvNet::processDataset(SpatiallySparseDataset &dataset,
                                    int batchSize, float learningRate,
                                    float momentum) {
  return cnn->processDataset(dataset, batchSize, learningRate, momentum);
}

void SparseConvNet::processDatasetRepeatTest(
    SpatiallySparseDataset &dataset, int batchSize, int nReps,
    std::string predictionsFilename, std::string confusionMatrixFilename) {
  cnn->processDatasetRepeatTest(dataset, batchSize, nReps, predictionsFilename,
                                confusionMatrixFilename);
}

float SparseConvNet::processIndexLearnerDataset(SpatiallySparseDataset &dataset,
                                                int batchSize,
                                                float learningRate,
                                                float momentum) {
  return cnn->processIndexLearnerDataset(dataset, batchSize, learningRate,
                                         momentum);
}

void SparseConvNet::processDatasetDumpTopLevelFeatures(
    SpatiallySparseDataset &dataset, int batchSize, int reps) {
  cnn->processDatasetDumpTopLevelFeatures(dataset, batchSize, reps);
}

void SparseConvNet::loadWeights(std::string baseName, int epoch, bool momentum,
                                int firstNlayers) {
  cnn->loadWeights(baseName, epoch, momentum, firstNlayers);
}

void SparseConvNet::saveWeights(std::string baseName, int epoch,
                                bool momentum) {
  cnn->saveWeights(baseName, epoch, momentum);
}

void SparseConvNet::calculateInputRegularizingConstants(
    SpatiallySparseDataset dataset) {
  cnn->calculateInputRegularizingConstants(dataset);
}

SparseConvTriangLeNet::SparseConvTriangLeNet(int dimension, int nInputFeatures,
                                             int nClasses, int pciBusID,
                                             int nTop)
    : cnn(new SparseConvNetCUDA(dimension, nInputFeatures, nClasses, pciBusID,
                                nTop)) {}
SparseConvTriangLeNet::~SparseConvTriangLeNet() {}

void SparseConvTriangLeNet::addLeNetLayerMP(int nFeatures, int filterSize,
                                            int filterStride, int poolSize,
                                            int poolStride,
                                            ActivationFunction activationFn,
                                            float dropout,
                                            int minActiveInputs) {
  cnn->addTriangularLeNetLayerMP(nFeatures, filterSize, filterStride, poolSize,
                                 poolStride, activationFn, dropout,
                                 minActiveInputs);
}

void SparseConvTriangLeNet::addTerminalPoolingLayer(int poolSize) {
  cnn->addTerminalPoolingLayer(poolSize,
                               triangleSize(poolSize, cnn->dimension));
}

void SparseConvTriangLeNet::addSoftmaxLayer() { cnn->addSoftmaxLayer(); }

void SparseConvTriangLeNet::addIndexLearnerLayer() {
  cnn->addIndexLearnerLayer();
}

float SparseConvTriangLeNet::processDataset(SpatiallySparseDataset &dataset,
                                            int batchSize, float learningRate,
                                            float momentum) {
  return cnn->processDataset(dataset, batchSize, learningRate, momentum);
}

void SparseConvTriangLeNet::processDatasetRepeatTest(
    SpatiallySparseDataset &dataset, int batchSize, int nReps,
    std::string predictionsFilename, std::string confusionMatrixFilename) {
  cnn->processDatasetRepeatTest(dataset, batchSize, nReps, predictionsFilename,
                                confusionMatrixFilename);
}

float SparseConvTriangLeNet::processIndexLearnerDataset(
    SpatiallySparseDataset &dataset, int batchSize, float learningRate,
    float momentum) {
  return cnn->processIndexLearnerDataset(dataset, batchSize, learningRate,
                                         momentum);
}

void SparseConvTriangLeNet::processDatasetDumpTopLevelFeatures(
    SpatiallySparseDataset &dataset, int batchSize, int reps) {
  cnn->processDatasetDumpTopLevelFeatures(dataset, batchSize, reps);
}

void SparseConvTriangLeNet::loadWeights(std::string baseName, int epoch,
                                        bool momentum, int firstNlayers) {
  cnn->loadWeights(baseName, epoch, momentum, firstNlayers);
}

void SparseConvTriangLeNet::saveWeights(std::string baseName, int epoch,
                                        bool momentum) {
  cnn->saveWeights(baseName, epoch, momentum);
}

void SparseConvTriangLeNet::calculateInputRegularizingConstants(
    SpatiallySparseDataset dataset) {
  cnn->calculateInputRegularizingConstants(dataset);
}
