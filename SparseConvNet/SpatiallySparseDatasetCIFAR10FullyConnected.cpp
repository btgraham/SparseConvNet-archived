#include "SpatiallySparseDatasetCIFAR10FullyConnected.h"
#include <vector>
#include <iostream>
#include <fstream>

class TrivialPicture : public Picture {
public:
  std::vector<float> features;
  TrivialPicture(int label = -1) : Picture(label) {}
  ~TrivialPicture() {}
  Picture *distort(RNG &rng, batchType type = TRAINBATCH) {
    TrivialPicture *pic = new TrivialPicture(*this);
    if (type == TRAINBATCH)
      for (int i = 0; i < features.size(); ++i)
        pic->features[i] *= rng.bernoulli(0.8);
    else
      for (int i = 0; i < features.size(); ++i)
        pic->features[i] *= 0.8;
    return pic;
  };
  void codifyInputData(SparseGrid &grid, std::vector<float> &f,
                       int &nSpatialSites, int spatialSize) {
    grid.mp[0] = nSpatialSites++;
    for (int i = 0; i < features.size(); ++i)
      f.push_back(features[i]);
  }
};

void readCIFAR10File(std::vector<Picture *> &characters, const char *filename) {
  std::ifstream file(filename, std::ios::in | std::ios::binary);
  if (!file) {
    std::cout << "Cannot find " << filename << std::endl;
    exit(EXIT_FAILURE);
  }
  unsigned char label;
  while (file.read((char *)&label, 1)) {
    TrivialPicture *character = new TrivialPicture(label);
    unsigned char bitmap[3072];
    file.read((char *)bitmap, 3072);
    for (int x = 0; x < 3072; x++) {
      character->features.push_back(bitmap[x] / 127.5 - 1);
    }
    characters.push_back(character);
  }
  file.close();
}
SpatiallySparseDataset Cifar10TrainSetFullyConnected() {
  SpatiallySparseDataset dataset;
  dataset.name = "CIFAR-10 train set Fully connected";
  dataset.type = TRAINBATCH;
  dataset.nFeatures = 3072;
  dataset.nClasses = 10;
  char filenameFormat[] = "Data/CIFAR10/data_batch_%d.bin";
  char filename[100];
  for (int fileNumber = 1; fileNumber <= 5; fileNumber++) {
    sprintf(filename, filenameFormat, fileNumber);
    readCIFAR10File(dataset.pictures, filename);
  }
  return dataset;
}
SpatiallySparseDataset Cifar10TestSetFullyConnected() {
  SpatiallySparseDataset dataset;
  dataset.name = "CIFAR-10 test set fully connected";
  dataset.type = TESTBATCH;
  dataset.nFeatures = 3072;
  dataset.nClasses = 10;
  char filenameTest[] = "Data/CIFAR10/test_batch.bin";
  readCIFAR10File(dataset.pictures, filenameTest);
  return dataset;
}
