#include "SpatiallySparseDatasetCIFAR10.h"
#include <vector>
#include <iostream>
#include <fstream>

void readCIFAR10File(std::vector<Picture *> &characters, const char *filename) {
  std::ifstream file(filename, std::ios::in | std::ios::binary);
  if (!file) {
    std::cout << "Cannot find " << filename << std::endl;
    exit(EXIT_FAILURE);
  }
  unsigned char label;
  while (file.read((char *)&label, 1)) {
    OpenCVPicture *character = new OpenCVPicture(32, 32, 3, 128, label);
    unsigned char bitmap[3072];
    float *matData = ((float *)(character->mat.data));
    file.read((char *)bitmap, 3072);
    for (int x = 0; x < 32; x++) {
      for (int y = 0; y < 32; y++) {
        for (int c = 0; c < 3; c++) {
          matData[y * 96 + x * 3 + (2 - c)] = bitmap[c * 1024 + y * 32 + x];
        }
      }
    }
    characters.push_back(character);
  }
  file.close();
}
SpatiallySparseDataset Cifar10TrainSet() {
  SpatiallySparseDataset dataset;
  dataset.name = "CIFAR-10 train set";
  dataset.type = TRAINBATCH;
  dataset.nFeatures = 3;
  dataset.nClasses = 10;
  char filenameFormat[] = "Data/CIFAR10/data_batch_%d.bin";
  char filename[100];
  for (int fileNumber = 1; fileNumber <= 5; fileNumber++) {
    sprintf(filename, filenameFormat, fileNumber);
    readCIFAR10File(dataset.pictures, filename);
  }
  return dataset;
}
SpatiallySparseDataset Cifar10TestSet() {
  SpatiallySparseDataset dataset;
  dataset.name = "CIFAR-10 test set";
  dataset.type = TESTBATCH;
  dataset.nFeatures = 3;
  dataset.nClasses = 10;
  char filenameTest[] = "Data/CIFAR10/test_batch.bin";
  readCIFAR10File(dataset.pictures, filenameTest);
  return dataset;
}
