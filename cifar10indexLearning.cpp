#include "SparseConvNet.h"
#include "SpatiallySparseDatasetCIFAR10.h"

int epoch = 0;
int pciBusID = -1; // PCI bus ID, -1 for default GPU
int batchSize = 300;

Picture *OpenCVPicture::distort(RNG &rng, batchType type) {
  OpenCVPicture *pic = new OpenCVPicture(*this);
  if (type == TRAINBATCH)
    pic->jiggle(rng, 4);
  return pic;
}

class CNN : public SparseConvNet {
public:
  CNN(int dimension, int nInputFeatures, int nClasses, int pciBusID)
      : SparseConvNet(dimension, nInputFeatures, nClasses, pciBusID) {
    addLeNetLayerMP(32, 3, 1, 2, 2, VLEAKYRELU);
    addLeNetLayerMP(32, 2, 1, 2, 2, VLEAKYRELU);
    addLeNetLayerMP(32, 2, 1, 2, 2, VLEAKYRELU);
    addLeNetLayerMP(32, 2, 1, 1, 1, VLEAKYRELU);
    addLeNetLayerMP(32 * 2, 1, 1, 1, 1, NOSIGMOID, 0.5);
    addIndexLearnerLayer();
  }
};

int main() {
  std::string baseName = "weights/Cifar10IndexLearner";

  SpatiallySparseDataset trainSet = Cifar10TrainSet(); // For training
  SpatiallySparseDataset trainSet2 = Cifar10TrainSet();
  trainSet2.type = TESTBATCH; // For calculating the top hidden layer for the
                              // training set (without dropout or data
                              // augmentation)
  SpatiallySparseDataset testSet =
      Cifar10TestSet(); // For calculating the top hidden layer for the test set

  trainSet.summary();
  trainSet2.summary();
  testSet.summary();
  CNN cnn(2, trainSet.nFeatures, trainSet.pictures.size(), pciBusID);
  if (epoch > 0) {
    cnn.loadWeights(baseName, epoch);
  }
  for (epoch++; epoch <= 100000; epoch++) {
    std::cout << "epoch: " << epoch << std::endl;
    cnn.processIndexLearnerDataset(trainSet, batchSize, 0.0003);
    if (epoch % 20 == 0) {
      cnn.saveWeights(baseName, epoch);
      cnn.processIndexLearnerDataset(trainSet2, batchSize);
      cnn.processIndexLearnerDataset(testSet, batchSize);
    }
  }
  // // // {//After pretraining, convert to a regular neural network
  // // //
  // cnn.nOutputFeatures=dynamic_cast<IndexLearnerLayer*>(cnn.cnn.back())->nFeaturesIn;
  // // //   delete cnn.cnn.back();
  // // //   cnn.cnn.pop_back();
  // // //   cnn.nClasses=trainSet.nClasses;
  // // //   cnn.addLearntLayer(1024);cout <<endl;
  // // //   cnn.addLearntLayer(1024);cout <<endl;
  // // //   cnn.addSoftmaxLayer();cout <<endl;
  // // // }
  // // // for (;;epoch++) {
  // // //   cout <<"epoch: " << epoch << flush;
  // // //   cnn.processDataset(trainSet, batchSize,0.0003);
  // // //   if (epoch%10==0) {
  // // //     cnn.saveWeights(baseName,epoch);
  // // //     cnn.processDatasetRepeatTest(testSet, batchSize, 1);
  // // //   }
  // // // }
  //
}
