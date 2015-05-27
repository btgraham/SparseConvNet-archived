#pragma once
#include "SpatiallySparseDataset.h"
#include "OpenCVPicture.h"

class KagglePlanktonLabeledDataSet : public SpatiallySparseDataset {
public:
  std::map<std::string,int> classes;
  KagglePlanktonLabeledDataSet(std::string classesListFile, std::string dataDirectory, batchType type_, int backgroundCol);
};
class KagglePlanktonUnlabeledDataSet : public SpatiallySparseDataset {
public:
  std::map<std::string,int> classes;
  std::string header;
  KagglePlanktonUnlabeledDataSet(std::string classesListFile, std::string dataDirectory, int backgroundCol);
};
