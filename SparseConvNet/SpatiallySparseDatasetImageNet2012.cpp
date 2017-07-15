#include "SpatiallySparseDatasetImageNet2012.h"
#include "SpatiallySparseDatasetOpenCV.h"
#include <iostream>
#include <fstream>
#include <string>

SpatiallySparseDataset ImageNet2012TrainSet() {
  auto dataset = OpenCVLabeledDataSet("Data/imagenet2012/classList",
                                      "Data/imagenet2012/ILSVRC2012_img_train",
                                      "*.JPEG", TRAINBATCH, 128, false, 1);
  dataset.name = "ImageNet2012 train set";
  return dataset;
}
SpatiallySparseDataset ImageNet2012ValidationSet() {
  SpatiallySparseDataset dataset;
  dataset.name = "ImageNet2012 validation set";
  dataset.type = TESTBATCH;
  dataset.nFeatures = 3;
  dataset.nClasses = 1000;

  std::string imageFile;
  int cl;
  std::ifstream file("Data/imagenet2012/validationData.txt");
  int nWidth, nHeight, nBBoxx, nBBoxX, nBBoxy, nBBoxY;
  while (file >> cl >> imageFile >> nWidth >> nHeight >> nBBoxx >> nBBoxX >>
         nBBoxy >> nBBoxY) {
    cl--;
    OpenCVPicture *pic = new OpenCVPicture(
        std::string("Data/imagenet2012/") + imageFile, 128, cl);
    dataset.pictures.push_back(pic);
  }
  return dataset;
}
SpatiallySparseDataset ImageNet2012TestSet() {
  SpatiallySparseDataset dataset;
  dataset.name = "ImageNet2012 train set";
  dataset.type = UNLABELEDBATCH;
  dataset.nFeatures = 3;
  dataset.nClasses = 1000;

  std::string imageFile;
  int cl;
  std::ifstream file("Data/imagenet2012/testData.txt");
  int nWidth, nHeight, nBBoxx, nBBoxX, nBBoxy, nBBoxY;
  while (file >> cl >> imageFile >> nWidth >> nHeight >> nBBoxx >> nBBoxX >>
         nBBoxy >> nBBoxY) {
    OpenCVPicture *pic = new OpenCVPicture(
        std::string("Data/imagenet2012/") + imageFile, 128, -1);
    dataset.pictures.push_back(pic);
  }
  return dataset;
}
