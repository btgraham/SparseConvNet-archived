// Try adding normal vectors to input data.

// Data from http://www.icst.pku.edu.cn/zlian/shrec15-non-rigid/index.htm
// 50 classes, 24 exemplars per class: alien ants armadillo bird1 bird2 camel
// cat centaur twoballs dinosaur dog1 dog2 glasses gorilla hand horse lamp paper
// man octopus pliers rabbit santa scissor shark snake spider dino_ske flamingo
// woman Aligator Bull Chick Deer Dragon Elephant Frog Giraffe Kangaroo Mermaid
// Mouse Nunchaku MantaRay Ring Robot Sumotori Tortoise Watch Weedle Woodman

#include "SpatiallySparseDatasetSHREC2015.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include "OpenCVPicture.h"
#include "Off3DFormatPicture.h"

SpatiallySparseDataset SHREC2015TrainSet(int renderSize, int kFold, int fold) {
  SpatiallySparseDataset dataset;
  dataset.name = "SHREC2015-Non-Rigid (Train subset)";
  dataset.type = TRAINBATCH;
  dataset.nFeatures = 1;
  dataset.nClasses = 50;
  std::ifstream cla("Data/SHREC15/SHREC15_Non-rigid_ToolKit/test.cla");
  std::string line;
  int nClasses, nTotal;
  getline(cla, line); // header line
  cla >> nClasses >> nTotal;
  for (int cl = 0; cl < nClasses; cl++) {
    getline(cla, line); // blank line
    std::string className;
    int parent, nExemplars;
    cla >> className >> parent >> nExemplars;
    for (int exemp = 0; exemp < nExemplars; exemp++) {
      int num;
      cla >> num;
      std::string filename =
          std::string("Data/SHREC15/SHREC15NonRigidTestDB/T") +
          std::to_string(num) + std::string(".off");
      if (exemp % kFold != fold)
        dataset.pictures.push_back(
            new OffSurfaceModelPicture(filename, renderSize, cl));
    }
  }
  return dataset;
};

SpatiallySparseDataset SHREC2015TestSet(int renderSize, int kFold, int fold) {
  SpatiallySparseDataset dataset;
  dataset.name = "SHREC2015-Non-Rigid (Validation subset)";
  dataset.type = TESTBATCH;
  dataset.nFeatures = 1;
  dataset.nClasses = 50;
  std::ifstream cla("Data/SHREC15/SHREC15_Non-rigid_ToolKit/test.cla");
  std::string line;
  int nClasses, nTotal;
  getline(cla, line); // header line
  cla >> nClasses >> nTotal;
  for (int cl = 0; cl < nClasses; cl++) {
    getline(cla, line); // blank line
    std::string className;
    int parent, nExemplars;
    cla >> className >> parent >> nExemplars;
    for (int exemp = 0; exemp < nExemplars; exemp++) {
      int num;
      cla >> num;
      std::string filename =
          std::string("Data/SHREC15/SHREC15NonRigidTestDB/T") +
          std::to_string(num) + std::string(".off");
      if (exemp % kFold == fold)
        dataset.pictures.push_back(
            new OffSurfaceModelPicture(filename, renderSize, cl));
    }
  }
  return dataset;
};
