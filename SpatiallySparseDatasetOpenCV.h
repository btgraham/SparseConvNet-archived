#pragma once
#include "SpatiallySparseDataset.h"
#include "OpenCVPicture.h"

class OpenCVLabeledDataSet : public SpatiallySparseDataset {
public:
  std::map<std::string,int> classes;
  OpenCVLabeledDataSet(std::string classesListFile, std::string dataDirectory, std::string wildcard,
                       batchType type_, int backgroundCol=128,
                       bool loadData=true, int flags=1);
};
class OpenCVUnlabeledDataSet : public SpatiallySparseDataset {
public:
  std::map<std::string,int> classes;
  std::string header;
  OpenCVUnlabeledDataSet(std::string classesListFile, std::string dataDirectory, std::string wildcard,
                         int backgroundCol=128,
                         bool loadData=true, int flags=1);
};
