#include "SpatiallySparseDatasetImageNet2012.h"
#include<iostream>
#include<fstream>
#include<string>
SpatiallySparseDataset ImageNet2012TrainSet(int scale,int n) {
  SpatiallySparseDataset dataset;
  dataset.name="ImageNet2012 train set";
  dataset.type=TRAINBATCH;
  dataset.nFeatures=3;
  dataset.nClasses=1000;

  std::string imageFile;
  int cl;
  std::ifstream file("Data/imagenet2012/trainingData.txt");
  std::vector<int> count(1000);
  int nWidth, nHeight, nBBoxx, nBBoxX, nBBoxy, nBBoxY;
  while (file >> cl >> imageFile >> nWidth >> nHeight >> nBBoxx >> nBBoxX >> nBBoxy >> nBBoxY) {
    cl--;
    if (count[cl]++<n) {
      OpenCVPicture*  pic = new OpenCVPicture(std::string("Data/imagenet2012/")+imageFile,scale,128,cl);
      dataset.pictures.push_back(pic);
    }
  }
  return dataset;
}
SpatiallySparseDataset ImageNet2012ValidationSet(int scale) {
  SpatiallySparseDataset dataset;
  dataset.name="ImageNet2012 validation set";
  dataset.type=TESTBATCH;
  dataset.nFeatures=3;
  dataset.nClasses=1000;

  std::string imageFile;
  int cl;
  std::ifstream file("Data/imagenet2012/validationData.txt");
  int nWidth, nHeight, nBBoxx, nBBoxX, nBBoxy, nBBoxY;
  while (file >> cl >> imageFile >> nWidth >> nHeight >> nBBoxx >> nBBoxX >> nBBoxy >> nBBoxY) {
    cl--;
    OpenCVPicture*  pic = new OpenCVPicture(std::string("Data/imagenet2012/")+imageFile,scale,128,cl);
    dataset.pictures.push_back(pic);
  }
  return dataset;
}
SpatiallySparseDataset ImageNet2012TestSet(int scale) {
  SpatiallySparseDataset dataset;
  dataset.name="ImageNet2012 train set";
  dataset.type=UNLABELEDBATCH;
  dataset.nFeatures=3;
  dataset.nClasses=1000;

  std::string imageFile;
  int cl;
  std::ifstream file("Data/imagenet2012/testData.txt");
  int nWidth, nHeight, nBBoxx, nBBoxX, nBBoxy, nBBoxY;
  while (file >> cl >> imageFile >> nWidth >> nHeight >> nBBoxx >> nBBoxX >> nBBoxy >> nBBoxY) {
    OpenCVPicture*  pic = new OpenCVPicture(std::string("Data/imagenet2012/")+imageFile,scale,128,-1);
    dataset.pictures.push_back(pic);
  }
  return dataset;
}
