// http://www.nada.kth.se/cvap/actions/

#include "SpatiallySparseDatasetCVAP_RHA.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include "CVAP_RHA_Picture.h"

SpatiallySparseDataset CVAP_RHA_TrainSet() {
  SpatiallySparseDataset dataset;
  dataset.name = "CVAP-RHA Trainset";
  dataset.type = TRAINBATCH;
  dataset.nFeatures = 1;
  dataset.nClasses = 6;
  std::ifstream d("Data/CVAP_RHA/train.dataset");
  assert(d);
  while (!d.eof()) {
    try {
      dataset.pictures.push_back(new CVAP_RHA_Picture(d));
    } catch (int e) {
    }
  }
  return dataset;
};
SpatiallySparseDataset CVAP_RHA_ValidationSet() {
  SpatiallySparseDataset dataset;
  dataset.name = "CVAP-RHA Validationset";
  dataset.type = TESTBATCH;
  dataset.nFeatures = 1;
  dataset.nClasses = 6;
  std::ifstream d("Data/CVAP_RHA/validation.dataset");
  while (!d.eof()) {
    try {
      dataset.pictures.push_back(new CVAP_RHA_Picture(d));
    } catch (int e) {
    }
  }
  return dataset;
};
SpatiallySparseDataset CVAP_RHA_TestSet() {
  SpatiallySparseDataset dataset;
  dataset.name = "CVAP-RHA Testset";
  dataset.type = TESTBATCH;
  dataset.nFeatures = 1;
  dataset.nClasses = 6;
  std::ifstream d("Data/CVAP_RHA/test.dataset");
  while (!d.eof()) {
    try {
      dataset.pictures.push_back(new CVAP_RHA_Picture(d));
    } catch (int e) {
    }
  }
  return dataset;
};
