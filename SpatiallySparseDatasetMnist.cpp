#include "SpatiallySparseDatasetMnist.h"
#include <vector>
#include <iostream>
#include <fstream>

static int intToggleEndianness(int a) {
  int b = 0;
  b += a % 256 * (1 << 24);
  a >>= 8;
  b += a % 256 * (1 << 16);
  a >>= 8;
  b += a % 256 * (1 << 8);
  a >>= 8;
  b += a % 256 * (1 << 0);
  return b;
}

static void loadMnistC(std::string filename,
                       std::vector<Picture *> &characters) {
  std::ifstream f(filename.c_str());
  if (!f) {
    std::cout << "Cannot find " << filename << std::endl;
    ;
    exit(EXIT_FAILURE);
  }
  int a, n1, n2, n3;
  f.read((char *)&a, 4);
  f.read((char *)&a, 4);
  n1 = intToggleEndianness(a);
  f.read((char *)&a, 4);
  n2 = intToggleEndianness(a);
  f.read((char *)&a, 4);
  n3 = intToggleEndianness(a);
  std::vector<unsigned char> bitmap(n2 * n3);
  for (int i1 = 0; i1 < n1; i1++) {
    f.read((char *)&bitmap[0], n2 * n3);
    OpenCVPicture *character = new OpenCVPicture(n2, n3, 1, 0);
    float *matData = ((float *)(character->mat.data));
    for (int i = 0; i < n2 * n3; i++)
      matData[i] = bitmap[i];
    characters.push_back(character);
  }
}
static void loadMnistL(std::string filename,
                       std::vector<Picture *> &characters) {
  std::ifstream f(filename.c_str());
  if (!f) {
    std::cout << "Cannot find " << filename << std::endl;
    exit(EXIT_FAILURE);
  }
  int a, n;
  char l;
  f.read((char *)&a, 4);
  f.read((char *)&a, 4);
  n = intToggleEndianness(a);
  for (int i = 0; i < n; i++) {
    f.read(&l, 1);
    characters[i]->label = l;
  }
}

SpatiallySparseDataset MnistTrainSet() {
  SpatiallySparseDataset dataset;
  dataset.name = "MNIST train set";
  dataset.type = TRAINBATCH;
  dataset.nFeatures = 1;
  dataset.nClasses = 10;
  std::string trainC("Data/MNIST/train-images-idx3-ubyte");
  std::string trainL("Data/MNIST/train-labels-idx1-ubyte");
  loadMnistC(trainC, dataset.pictures);
  loadMnistL(trainL, dataset.pictures);
  return dataset;
}
SpatiallySparseDataset MnistTestSet() {
  SpatiallySparseDataset dataset;
  dataset.type = TESTBATCH;
  dataset.name = "MNIST test set";
  dataset.nFeatures = 1;
  dataset.nClasses = 10;
  std::string testC("Data/MNIST/t10k-images-idx3-ubyte");
  std::string testL("Data/MNIST/t10k-labels-idx1-ubyte");
  loadMnistC(testC, dataset.pictures);
  loadMnistL(testL, dataset.pictures);
  return dataset;
}
