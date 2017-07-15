#include "SparseConvNet.h"
#include "SpatiallySparseDatasetUCF101.h"

int epoch = 0;
int cudaDevice = -1;
int batchSize = 1;

class CNN : public SparseConvNet {
public:
  CNN(int dimension, int l, ActivationFunction fn, int nInputFeatures,
      int nClasses, float p = 0.0f, int cudaDevice = -1, int nTop = 1)
      : SparseConvNet(dimension, nInputFeatures, nClasses, cudaDevice, nTop) {
    for (int i = 0; i <= l; i++)
      addLeNetLayerMP(32 * (i + 1) + 32 * std::max(i - 3, 0), ///////////////
                      2, 1, (i < l) ? 3 : 1, (i < l) ? 2 : 1, fn,
                      p * i * 1.0f / l);
    addSoftmaxLayer();
  }
};

int main(int lenArgs, char *args[]) {
  std::string baseName = "weights/UCF101";
  SpatiallySparseDataset trainSet = UCF101TrainSet();
  SpatiallySparseDataset testSet = UCF101TestSet();
  trainSet.summary();
  testSet.summary();

  CNN cnn(3, 6, VLEAKYRELU, trainSet.nFeatures, trainSet.nClasses, 0,
          cudaDevice);
  if (epoch > 0) {
    cnn.loadWeights(baseName, epoch);
    cnn.processDatasetRepeatTest(testSet, batchSize, 12);
  }
  for (epoch++;; epoch++) {
    std::cout << "epoch:" << epoch << ": " << std::flush;
    for (int i = 0; i < 5; ++i) {
      cnn.processDataset(trainSet, batchSize, 0.003 * exp(-0.05 * epoch),
                         0.9999);
      cnn.saveWeights(baseName, epoch);
    }
    cnn.processDatasetRepeatTest(testSet, batchSize, 1);
  }
}
