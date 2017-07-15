#include "SparseConvNet.h"
#include "SpatiallySparseDatasetSHREC2015.h"

int epoch = 0;
int cudaDevice = -1;
int batchSize = 10;

class DeepC2Triangular : public SparseConvNet {
public:
  DeepC2Triangular(int dimension, int l, int k, ActivationFunction fn,
                   int nInputFeatures, int nClasses, float p = 0.0f,
                   int cudaDevice = -1, int nTop = 1);
};
DeepC2Triangular::DeepC2Triangular(int dimension, int l, int k,
                                   ActivationFunction fn, int nInputFeatures,
                                   int nClasses, float p, int cudaDevice,
                                   int nTop)
    : SparseConvNet(dimension, nInputFeatures, nClasses, cudaDevice, nTop) {
  for (int i = 0; i <= l; i++)
    addTriangularLeNetLayerMP((i + 1) * k, (i == l) ? 2 : 2, 1, (i < l) ? 3 : 1,
                              (i < l) ? 2 : 1, fn, p * i * 1.0f / l);
  addSoftmaxLayer();
}

int main(int lenArgs, char *args[]) {
  std::string baseName = "weights/SHREC2015";
  int fold = atoi(args[1]);
  std::cout << "Fold: " << fold << std::endl;
  SpatiallySparseDataset trainSet = SHREC2015TrainSet(80, 6, fold);
  trainSet.summary();
  trainSet.repeatSamples(10);
  SpatiallySparseDataset testSet = SHREC2015TestSet(80, 6, fold);
  testSet.summary();

  DeepC2Triangular cnn(3, 6, 32, VLEAKYRELU, trainSet.nFeatures,
                       trainSet.nClasses, 0.0f, cudaDevice);
  if (epoch > 0)
    cnn.loadWeights(baseName, epoch);
  for (epoch++; epoch <= 100 * 2; epoch++) {
    std::cout << "epoch:" << epoch << ": " << std::flush;
    cnn.processDataset(trainSet, batchSize, 0.003 * exp(-0.05 / 2 * epoch));
    if (epoch % 20 == 0) {
      cnn.saveWeights(baseName, epoch);
      cnn.processDatasetRepeatTest(testSet, batchSize, 3);
    }
  }
}
