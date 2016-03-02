#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <iostream>
#include "gbcodes3755.h"
#include "OnlineHandwritingPicture.h"
#include "SpatiallySparseDatasetPenDigits.h"

struct iPoint {
  short x;
  short y;
};

void readPenDigitsFile(std::vector<Picture *> &characters, const char *filename,
                       int renderSize, OnlineHandwritingEncoding enc) {
  std::ifstream f(filename, std::ios::in | std::ios::binary);
  if (!f) {
    std::cout << "Cannot find " << filename << std::endl;
    std::cout << "Please download it from the UCI Machine Learning Repository:"
              << std::endl;
    std::cout << "http://archive.ics.uci.edu/ml/datasets/"
                 "Pen-Based+Recognition+of+Handwritten+Digits" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::vector<std::string> data;
  std::copy(std::istream_iterator<std::string>(f),
            std::istream_iterator<std::string>(), std::back_inserter(data));
  int pen = 0;
  OnlineHandwritingPicture *character =
      new OnlineHandwritingPicture(renderSize, enc, -1);
  std::vector<iPoint> stroke;
  for (unsigned int i = 0; i < data.size(); i++) {
    if (data[i] == ".COMMENT") {
      if (character->ops.size() > 0) {
        character->normalize();
        characters.push_back(character);
        character = new OnlineHandwritingPicture(renderSize, enc, -1);
      }
      character->label = atoi(data[i + 1].c_str());
    }
    if (data[i] == ".PEN_UP") {
      pen = 0;
      character->ops.push_back(arma::mat(stroke.size(), 2));
      for (unsigned int i = 0; i < stroke.size(); ++i) {
        character->ops.back()(i, 0) = stroke[i].x;
        character->ops.back()(i, 1) = stroke[i].y;
      }
      stroke.clear();
    }
    if (pen == 1) {
      iPoint p;
      p.y = atoi(data[i].c_str());
      i++;
      p.x = -atoi(data[i].c_str());
      stroke.push_back(p);
    }
    if (data[i] == ".PEN_DOWN")
      pen = 1;
  }
  character->normalize();
  characters.push_back(character);
}

SpatiallySparseDataset PenDigitsTrainSet(int renderSize,
                                         OnlineHandwritingEncoding enc) {
  SpatiallySparseDataset dataset;
  dataset.name = "Pendigits train set";
  dataset.type = TRAINBATCH;
  dataset.nFeatures = OnlineHandwritingEncodingSize[enc];
  dataset.nClasses = 10;
  readPenDigitsFile(dataset.pictures, "Data/pendigits/pendigits-orig.tra",
                    renderSize, enc);
  return dataset;
}
SpatiallySparseDataset PenDigitsTestSet(int renderSize,
                                        OnlineHandwritingEncoding enc) {
  SpatiallySparseDataset dataset;
  dataset.name = "Pendigits test set";
  dataset.type = TESTBATCH;
  dataset.nFeatures = OnlineHandwritingEncodingSize[enc];
  dataset.nClasses = 10;
  readPenDigitsFile(dataset.pictures, "Data/pendigits/pendigits-orig.tes",
                    renderSize, enc);
  return dataset;
}
