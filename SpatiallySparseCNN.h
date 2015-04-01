#pragma once

class SpatiallySparseCNN {
public:
  vector<SpatiallySparseLayer*> cnn;
  int nInputFeatures;
  int nOutputFeatures;
  int inputSpatialSize;
  int nClasses;
  int nTop;
  SpatiallySparseCNN (int nInputFeatures,
                      int nClasses,
                      int cudaDevice=0,
                      int nTop=1) :
    nInputFeatures(nInputFeatures),
    nClasses(nClasses),
    nTop(nTop) {
    nOutputFeatures=nInputFeatures;
    initializeGPU(cudaDevice);
  }
  //Could apply the sigmoid after pooling when doing pooling
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
  void addXConvLayer(int nFeatures,
                     int filterSize,
                     int filterStride,
                     ActivationFunction activationFn=RELU,
                     float dropout=0.0f,
                     float poolingToFollow=1.0f);
  void addX4ConvLayer(int nFeatures,
                      ActivationFunction activationFn=RELU);
  void addYConvLayer(int nFeatures,
                     int filterStride=1,
                     ActivationFunction activationFn=RELU,
                     float dropout=0.0f,
                     float poolingToFollow=1.0f);
  void addLeNetLayerMP(int nFeatures, int filterSize, int filterStride, int poolSize, int poolStride, ActivationFunction activationFn=RELU, float dropout=0.0f);
  void addLeNetLayerAP(int nFeatures, int filterSize, int filterStride, int poolSize, int poolStride, ActivationFunction activationFn=RELU, float dropout=0.0f);
  void addLeNetLayerROFMP(int nFeatures, int filterSize, int filterStride, float fmpShrink, ActivationFunction activationFn=RELU, float dropout=0.0f);
  void addLeNetLayerPOFMP(int nFeatures, int filterSize, int filterStride, float fmpShrink, ActivationFunction activationFn=RELU, float dropout=0.0f);
  void addLeNetLayerJOFMP(int nFeatures, int filterSize, int filterStride, float fmpShrink, int poolSize, ActivationFunction activationFn=RELU, float dropout=0.0f);
  void addSoftmaxLayer();
  void addIndexLearnerLayer();
  void processBatch(SpatiallySparseBatch& batch, float learningRate, ofstream& f, ofstream& g);
  float processDataset(SpatialDataset &dataset, int batchSize=100, float learningRate=0);
  void processDatasetRepeatTest(SpatialDataset &dataset, int batchSize=100, int nReps=12, string predictionsFilename="",string header="");
  void loadWeights(string baseName, int epoch, int firstNlayers=1000000);
  void saveWeights(string baseName, int epoch);
  void processIndexLearnerBatch(SpatiallySparseBatch& batch, float learningRate, ofstream& f);
  float processIndexLearnerDataset(SpatialDataset &dataset, int batchSize=100, float learningRate=0.1);
  void processBatchDumpTopLevelFeaturess(SpatiallySparseBatch& batch, ofstream& f);
  void processDatasetDumpTopLevelFeatures(SpatialDataset &dataset, int batchSize=100);
};
