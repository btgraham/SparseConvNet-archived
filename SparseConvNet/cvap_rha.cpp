#include "SparseConvNet.h"
#include "SpatiallySparseDatasetCVAP_RHA.h"

int epoch = 0;
int cudaDevice = -1;
int batchSize = 1;

class CNN : public SparseConvNet {
public:
  CNN(int dimension, int l, ActivationFunction fn, int nInputFeatures,
      int nClasses, float p = 0.0f, int cudaDevice = -1, int nTop = 1)
      : SparseConvNet(dimension, nInputFeatures, nClasses, cudaDevice, nTop) {
    for (int i = 0; i <= l; i++)
      addLeNetLayerMP(32 * (i + 1) + 32 * std::max(i - 3, 0), 2, 1,
                      (i < l) ? 3 : 1, (i < l) ? 2 : 1, fn, p * i * 1.0f / l);
    addSoftmaxLayer();
  }
};

int main(int lenArgs, char *args[]) {
  std::string baseName = "weights/CVAP_RHA";
  SpatiallySparseDataset trainSet = CVAP_RHA_TrainSet();
  SpatiallySparseDataset validationSet = CVAP_RHA_ValidationSet();
  SpatiallySparseDataset testSet = CVAP_RHA_TestSet();
  trainSet.summary();
  validationSet.summary();
  testSet.summary();

  CNN cnn(3, 6, VLEAKYRELU, trainSet.nFeatures, trainSet.nClasses, 0.2,
          cudaDevice);
  if (epoch > 0) {
    cnn.loadWeights(baseName, epoch);
  }
  for (epoch++; epoch <= 100; epoch++) {
    std::cout << "epoch:" << epoch << ":\n";
    for (int i = 0; i < 20; ++i)
      cnn.processDataset(trainSet, batchSize, 0.003 * exp(-0.05 * epoch),
                         0.9999);
    cnn.saveWeights(baseName, epoch);
    cnn.processDatasetRepeatTest(validationSet, batchSize, 3);
  }
  cnn.processDatasetRepeatTest(testSet, batchSize, 3);
}
