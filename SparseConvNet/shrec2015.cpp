//#include "SparseConvNetCUDA.h"
#include "NetworkArchitectures.h"
#include "SpatiallySparseDatasetSHREC2015.h"

int epoch = 0;
int cudaDevice = -1;
int batchSize = 10;

int main(int lenArgs, char *args[]) {
  std::string baseName = "weights/SHREC2015";
  int fold = 0;
  if (lenArgs > 1)
    fold = atoi(args[1]);
  std::cout << "Fold: " << fold << std::endl;
  SpatiallySparseDataset trainSet = SHREC2015TrainSet(40, 6, fold);
  trainSet.summary();
  trainSet.repeatSamples(10);
  SpatiallySparseDataset testSet = SHREC2015TestSet(40, 6, fold);
  testSet.summary();

  DeepC2 cnn(3, 5, 32, VLEAKYRELU, trainSet.nFeatures, trainSet.nClasses, 0.0f,
             cudaDevice);
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
