#include "SparseConvNet.h"
#include "SparseConvNetCUDA.h"
#include "utilities.h"

SparseConvNet::SparseConvNet(int dimension, int nInputFeatures, int nClasses, int pciBusID, int nTop)
  : cnn(new SparseConvNetCUDA(dimension, nInputFeatures, nClasses, pciBusID, nTop)) {
}

SparseConvNet::~SparseConvNet(){ //Using auto_ptr for pimpl
}

void SparseConvNet::addLeNetLayerMP(int nFeatures, int filterSize, int filterStride, int poolSize, int poolStride, ActivationFunction activationFn, float dropout){
  cnn->addLeNetLayerMP(nFeatures, filterSize, filterStride, poolSize, poolStride, activationFn, dropout);
}

void SparseConvNet::addLeNetLayerPOFMP(int nFeatures, int filterSize, int filterStride, int poolSize, float fmpShrink, ActivationFunction activationFn, float dropout){
  cnn->addLeNetLayerPOFMP(nFeatures, filterSize, filterStride, poolSize, fmpShrink, activationFn, dropout);
}

void SparseConvNet::addLeNetLayerROFMP(int nFeatures, int filterSize, int filterStride, int poolSize, float fmpShrink, ActivationFunction activationFn, float dropout){
  cnn->addLeNetLayerROFMP(nFeatures, filterSize, filterStride, poolSize, fmpShrink, activationFn, dropout);
}

void SparseConvNet::addTerminalPoolingLayer(int poolSize){
  cnn->addTerminalPoolingLayer(poolSize, ipow(poolSize,cnn->dimension));
}

void SparseConvNet::addSoftmaxLayer(){
  cnn->addSoftmaxLayer();
}

void SparseConvNet::addIndexLearnerLayer(){
  cnn->addIndexLearnerLayer();
}

float SparseConvNet::processDataset(SpatiallySparseDataset &dataset, int batchSize, float learningRate) {
  return cnn->processDataset(dataset,batchSize,learningRate);
}

void SparseConvNet::processDatasetRepeatTest(SpatiallySparseDataset &dataset, int batchSize, int nReps, std::string predictionsFilename,std::string header){
  cnn->processDatasetRepeatTest(dataset,batchSize,nReps,predictionsFilename,header);
}

float SparseConvNet::processIndexLearnerDataset(SpatiallySparseDataset &dataset, int batchSize, float learningRate){
  return cnn->processIndexLearnerDataset(dataset,batchSize,learningRate);
}

void SparseConvNet::processDatasetDumpTopLevelFeatures(SpatiallySparseDataset &dataset, int batchSize, int reps){
  cnn->processDatasetDumpTopLevelFeatures(dataset,batchSize,reps);
}

void SparseConvNet::loadWeights(std::string baseName, int epoch, int firstNlayers){
  cnn->loadWeights(baseName,epoch,firstNlayers);
}

void SparseConvNet::saveWeights(std::string baseName, int epoch){
  cnn->saveWeights(baseName,epoch);
}

void SparseConvNet::calculateInputRegularizingConstants(SpatiallySparseDataset dataset){
  cnn->calculateInputRegularizingConstants(dataset);
}

SparseConvTriangLeNet::SparseConvTriangLeNet(int dimension, int nInputFeatures, int nClasses, int pciBusID, int nTop)
  : cnn(new SparseConvNetCUDA(dimension, nInputFeatures, nClasses, pciBusID, nTop)) {
}
SparseConvTriangLeNet::~SparseConvTriangLeNet(){ //Using auto_ptr for pimpl
}

void SparseConvTriangLeNet::addLeNetLayerMP(int nFeatures, int filterSize, int filterStride, int poolSize, int poolStride, ActivationFunction activationFn, float dropout){
  cnn->addTriangularLeNetLayerMP(nFeatures, filterSize, filterStride, poolSize, poolStride, activationFn, dropout);
}

void SparseConvTriangLeNet::addTerminalPoolingLayer(int poolSize) {
  cnn->addTerminalPoolingLayer(poolSize, triangleSize(poolSize,cnn->dimension));
}

void SparseConvTriangLeNet::addSoftmaxLayer(){
  cnn->addSoftmaxLayer();
}

void SparseConvTriangLeNet::addIndexLearnerLayer(){
  cnn->addIndexLearnerLayer();
}

float SparseConvTriangLeNet::processDataset(SpatiallySparseDataset &dataset, int batchSize, float learningRate){
  return cnn->processDataset(dataset,batchSize,learningRate);
}

void SparseConvTriangLeNet::processDatasetRepeatTest(SpatiallySparseDataset &dataset, int batchSize, int nReps, std::string predictionsFilename,std::string header){
  cnn->processDatasetRepeatTest(dataset,batchSize,nReps,predictionsFilename,header);
}

float SparseConvTriangLeNet::processIndexLearnerDataset(SpatiallySparseDataset &dataset, int batchSize, float learningRate){
  return cnn->processIndexLearnerDataset(dataset,batchSize,learningRate);
}

void SparseConvTriangLeNet::processDatasetDumpTopLevelFeatures(SpatiallySparseDataset &dataset, int batchSize, int reps){
  cnn->processDatasetDumpTopLevelFeatures(dataset,batchSize,reps);
}

void SparseConvTriangLeNet::loadWeights(std::string baseName, int epoch, int firstNlayers){
  cnn->loadWeights(baseName,epoch,firstNlayers);
}

void SparseConvTriangLeNet::saveWeights(std::string baseName, int epoch){
  cnn->saveWeights(baseName,epoch);
}

void SparseConvTriangLeNet::calculateInputRegularizingConstants(SpatiallySparseDataset dataset){
  cnn->calculateInputRegularizingConstants(dataset);
}
