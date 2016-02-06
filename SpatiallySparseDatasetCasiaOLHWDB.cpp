// Obtain POT files from
// http://www.nlpr.ia.ac.cn/databases/handwriting/home.html and put them in a
// directory Data/CASIA_pot_files/
#include <vector>
#include <fstream>
#include <algorithm>
#include <iostream>
#include "gbcodes3755.h"
#include "OnlineHandwritingPicture.h"
#include "SpatiallySparseDatasetCasiaOLHWDB.h"
const int nCharacters = 3755;

struct potCharacterHeader {
  unsigned short sampleSize;
  unsigned short label;
  unsigned short zzz;
  unsigned short nStrokes;
};

struct iPoint {
  short x;
  short y;
};

int readPotFile(std::vector<Picture *> &characters, const char *filename,
                int renderSize, OnlineHandwritingEncoding enc) {
  std::ifstream file(filename, std::ios::in | std::ios::binary);
  if (!file) {
    std::cout << "Cannot find " << filename << std::endl;
    exit(EXIT_FAILURE);
  }
  potCharacterHeader pCH;
  iPoint iP;
  int label;
  std::vector<iPoint> stroke;
  while (file.read((char *)&pCH, sizeof(potCharacterHeader))) {
    label = pCH.label;
    label = std::find(gbcodesPOT, gbcodesPOT + 3755, label) - gbcodesPOT;
    OnlineHandwritingPicture *character =
        new OnlineHandwritingPicture(renderSize, enc, label, 0.0001);
    file.read((char *)&iP, sizeof(iPoint));
    while (iP.y != -1) {
      stroke.resize(0);
      while (iP.x != -1) {
        stroke.push_back(iP);
        file.read((char *)&iP, sizeof(iPoint));
      }
      character->ops.push_back(arma::mat(stroke.size(), 2));
      for (unsigned int i = 0; i < stroke.size(); ++i) {
        character->ops.back()(i, 0) = stroke[i].x;
        character->ops.back()(i, 1) = stroke[i].y;
      }
      file.read((char *)&iP, sizeof(iPoint));
    }
    character->normalize();
    if (character->label < nCharacters)
      characters.push_back(character);
    else
      delete character;
  }
  file.close();
  return 0;
}

SpatiallySparseDataset CasiaOLHWDB11TrainSet(int renderSize,
                                             OnlineHandwritingEncoding enc) {
  SpatiallySparseDataset dataset;
  dataset.name = "CASIA OLHWDB1.1 train set--240 writers";
  dataset.type = TRAINBATCH;
  dataset.nFeatures = OnlineHandwritingEncodingSize[enc];
  dataset.nClasses = 3755;
  char filenameFormat[] = "Data/CASIA_pot_files/%04d.pot";
  char filename[100];
  for (int fileNumber = 1001; fileNumber <= 1240; fileNumber++) {
    sprintf(filename, filenameFormat, fileNumber);
    readPotFile(dataset.pictures, filename, renderSize, enc);
  }
  return dataset;
}
SpatiallySparseDataset
CasiaOLHWDB101112TrainSet(int renderSize, OnlineHandwritingEncoding enc) {
  SpatiallySparseDataset dataset;
  dataset.name = "CASIA OLHWDB1.0-1.2 train sets";
  dataset.type = TRAINBATCH;
  dataset.nFeatures = OnlineHandwritingEncodingSize[enc];
  dataset.nClasses = 3755;
  char filenameFormatA[] = "Data/CASIA_pot_files/%03d.pot";
  char filenameFormatB[] = "Data/CASIA_pot_files/%04d.pot";
  char filename[100];
  for (int fileNumber = 1; fileNumber <= 420; fileNumber++) {
    sprintf(filename, filenameFormatA, fileNumber);
    readPotFile(dataset.pictures, filename, renderSize, enc);
  }
  for (int fileNumber = 501; fileNumber <= 800; fileNumber++) {
    sprintf(filename, filenameFormatA, fileNumber);
    readPotFile(dataset.pictures, filename, renderSize, enc);
  }
  for (int fileNumber = 1001; fileNumber <= 1240; fileNumber++) {
    sprintf(filename, filenameFormatB, fileNumber);
    readPotFile(dataset.pictures, filename, renderSize, enc);
  }
  return dataset;
}
SpatiallySparseDataset CasiaOLHWDB11TestSet(int renderSize,
                                            OnlineHandwritingEncoding enc) {
  SpatiallySparseDataset dataset;
  dataset.name = "CASIA OLHWDB1.1 test set--60 writers";
  dataset.type = TESTBATCH;
  dataset.nFeatures = OnlineHandwritingEncodingSize[enc];
  dataset.nClasses = 3755;
  char filenameFormat[] = "Data/CASIA_pot_files/%04d.pot";
  char filename[100];
  for (int fileNumber = 1241; fileNumber <= 1300; fileNumber++) {
    sprintf(filename, filenameFormat, fileNumber);
    readPotFile(dataset.pictures, filename, renderSize, enc);
  }
  return dataset;
}
