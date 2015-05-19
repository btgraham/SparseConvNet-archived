#include "SpatiallySparseDatasetKagglePlankton.h"
#include<iostream>
#include<fstream>
#include<string>
SpatiallySparseDataset KagglePlanktonTrainSet() {
  SpatiallySparseDataset dataset;
  dataset.name="Kaggle Plankton train set";
  dataset.type=TRAINBATCH;
  dataset.nFeatures=1;
  dataset.nClasses=121;

  std::string imageFile;
  int cl;
  std::ifstream file("Data/__kagglePlankton/trainingData.txt");
  while (file >> cl >> imageFile) {
    OpenCVPicture* pic = new OpenCVPicture(std::string("Data/__kagglePlankton/")+imageFile,-1,0,cl);
    pic->loadDataOnceIgnoreScale();
    pic->scale=powf(powf(pic->mat.rows,2)+powf(pic->mat.cols,2),0.5);
    dataset.pictures.push_back(pic);
  }
  return dataset;
}
SpatiallySparseDataset KagglePlanktonTestSet() {
  SpatiallySparseDataset dataset;
  dataset.name="Kaggle Plankton test set";
  dataset.type=UNLABELLEDBATCH;
  dataset.nFeatures=1;
  dataset.nClasses=121;

  std::string imageFile;
  int cl;
  std::ifstream file("Data/__kagglePlankton/testData.txt");
  while (file >> cl >> imageFile) {
    OpenCVPicture* pic = new OpenCVPicture(std::string("Data/__kagglePlankton/test/")+imageFile,-1,0,cl*0);
    pic->loadDataOnceIgnoreScale();
    pic->scale=powf(powf(pic->mat.rows,2)+powf(pic->mat.cols,2),0.5);
    dataset.pictures.push_back(pic);
  }
  return dataset;
}
