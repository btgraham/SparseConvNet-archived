// http://crcv.ucf.edu/data/UCF101.php

#include "SpatiallySparseDatasetUCF101.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include "UCF101Picture.h"

SpatiallySparseDataset UCF101TrainSet() {
  SpatiallySparseDataset dataset;
  dataset.name = "UCF101 Trainset";
  dataset.type = TRAINBATCH;
  dataset.nFeatures = 3;
  dataset.nClasses = 101;
  std::ifstream d("Data/UCF101/trainlist01.dataset");
  while (!d.eof()) {
    try {
      dataset.pictures.push_back(new UCF101Picture(d));
    } catch (int e) {
    }
  }
  return dataset;
};
SpatiallySparseDataset UCF101TestSet() {
  SpatiallySparseDataset dataset;
  dataset.name = "UCF101 Testset";
  dataset.type = TESTBATCH;
  dataset.nFeatures = 3;
  dataset.nClasses = 101;
  std::ifstream d("Data/UCF101/testlist01.dataset");
  while (!d.eof()) {
    try {
      dataset.pictures.push_back(new UCF101Picture(d));
    } catch (int e) {
    }
  }
  return dataset;
};
