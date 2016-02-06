#include "SpatiallySparseDatasetOpenCV.h"
#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
#include <thread>
#include <fstream>
#include <iterator>

void loadDataThread(std::vector<Picture *> *pictures, int flags, unsigned int k,
                    int n) {
  for (; k < pictures->size(); k += n) {
    OpenCVPicture *pic = dynamic_cast<OpenCVPicture *>(pictures->at(k));
    // pic->loadDataWithoutScaling(flags);
    std::ifstream testFile(pic->filename.c_str(), std::ios::binary);
    pic->rawData = std::vector<char>((std::istreambuf_iterator<char>(testFile)),
                                     std::istreambuf_iterator<char>());
  }
}

OpenCVLabeledDataSet::OpenCVLabeledDataSet(std::string classesListFile,
                                           std::string dataDirectory,
                                           std::string wildcard,
                                           batchType type_, int backgroundCol,
                                           bool loadData, int flags) {
  name = dataDirectory;
  header = "image,label";
  type = type_;
  if (flags == 0)
    nFeatures = 1;
  if (flags == 1)
    nFeatures = 3;
  {
    std::ifstream f(classesListFile.c_str());
    std::string cl;
    int ctr = 0;
    while (f >> cl) {
      classes[cl] = ctr++;
      header += ',' + cl;
    }
  }
  nClasses = classes.size();
  for (auto &kv : classes) {
    for (auto &file :
         globVector(dataDirectory + "/" + kv.first + "/" + wildcard)) {
      OpenCVPicture *pic = new OpenCVPicture(file, backgroundCol, kv.second);
      pictures.push_back(pic);
    }
  }
  if (loadData) {
    std::vector<std::thread> workers;
    int nThreads = 4;
    for (int nThread = 0; nThread < nThreads; ++nThread)
      workers.emplace_back(loadDataThread, &pictures, flags, nThread, nThreads);
    for (int nThread = 0; nThread < nThreads; ++nThread)
      workers[nThread].join();
  }
}

OpenCVUnlabeledDataSet::OpenCVUnlabeledDataSet(std::string classesListFile,
                                               std::string dataDirectory,
                                               std::string wildcard,
                                               int backgroundCol, bool loadData,
                                               int flags) {
  name = dataDirectory;
  header = "image";
  type = UNLABELEDBATCH;
  if (flags == 0)
    nFeatures = 1;
  if (flags == 1)
    nFeatures = 3;
  {
    std::ifstream f(classesListFile.c_str());
    std::string cl;
    int ctr = 0;
    while (f >> cl) {
      classes[cl] = ctr++;
      header += "," + cl;
    }
  }
  nClasses = classes.size();
  for (auto &file : globVector(dataDirectory + "/" + wildcard)) {
    OpenCVPicture *pic = new OpenCVPicture(file, backgroundCol, 0);
    pictures.push_back(pic);
  }
  if (loadData) {
    std::vector<std::thread> workers;
    int nThreads = 4;
    for (int nThread = 0; nThread < nThreads; ++nThread)
      workers.emplace_back(loadDataThread, &pictures, flags, nThread, nThreads);
    for (int nThread = 0; nThread < nThreads; ++nThread)
      workers[nThread].join();
  }
}
