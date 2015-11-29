#include "SparseConvNet.h"
#include "SpatiallySparseDatasetSHREC2015.h"

int epoch = 0;
int cudaDevice = -1;
int batchSize = 10;

int nFeaturesPerLevel(int i) {
  return 32 * (i + 1); // This can be increased
}

class DeepPOFMP : public SparseConvNet {
public:
  DeepPOFMP(int dimension, int nInputFeatures, int nClasses, int cudaDevice,
            int nTop = 1)
      : SparseConvNet(dimension, nInputFeatures, nClasses, cudaDevice) {
    const float fmpShrink = powf(2, 0.6666); // Fractional Max-Pooling ratio.
    int l = 7; // Number of levels of FMP
    for (int i = 0; i < l; i++) {
      addLeNetLayerPOFMP(32 * (i + 1), 2, 1, 2, fmpShrink, VLEAKYRELU);
    }
    addLeNetLayerMP(32 * (l + 1), 2, 1, 1, 1, VLEAKYRELU);
    addLeNetLayerMP(32 * (l + 2), 1, 1, 1, 1, VLEAKYRELU);
    addSoftmaxLayer();
  }
};

int main(int lenArgs, char *args[]) {
  std::string baseName = "weights/SHREC2015_";
  int fold = 0;
  if (lenArgs > 1)
    fold = atoi(args[1]);
  std::cout << "Fold: " << fold << std::endl;
  SpatiallySparseDataset trainSet = SHREC2015TrainSet(32, 6, fold);
  trainSet.summary();
  trainSet.repeatSamples(10);
  SpatiallySparseDataset testSet = SHREC2015TestSet(32, 6, fold);
  testSet.summary();

  DeepPOFMP cnn(3, trainSet.nFeatures, trainSet.nClasses, cudaDevice);
  if (epoch > 0)
    cnn.loadWeights(baseName, epoch);
  for (epoch++; epoch <= 200; epoch++) {
    std::cout << "epoch: " << epoch << std::flush;
    cnn.processDataset(trainSet, batchSize, 0.003 * exp(-0.05 / 2 * epoch));
    if (epoch % 20 == 0) {
      cnn.saveWeights(baseName, epoch);
      cnn.processDatasetRepeatTest(testSet, batchSize, 3);
    }
  }
}
