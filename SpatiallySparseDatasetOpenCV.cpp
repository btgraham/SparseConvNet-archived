#include "SpatiallySparseDatasetOpenCV.h"
#include<iostream>
#include<fstream>
#include<string>
#include<cassert>
#include "utilities.h"

OpenCVLabeledDataSet::OpenCVLabeledDataSet
(std::string classesListFile, std::string dataDirectory, std::string wildcard,
 batchType type_, int backgroundCol,
 bool loadData, int flags) {
  name=dataDirectory;
  header="image,label";
  type=type_;
  if (flags==0) nFeatures=1;
  if (flags==1) nFeatures=3;
  {
    std::ifstream f(classesListFile.c_str());
    std::string cl;
    int ctr=0;
    while (f >> cl) {
      classes[cl]=ctr++;
      header+=','+cl;
    }
  }
  nClasses=classes.size();
  for (auto &kv : classes) {
    for (auto &file : globVector(dataDirectory+"/"+kv.first+"/"+wildcard)) {
      OpenCVPicture* pic = new OpenCVPicture(file,backgroundCol,kv.second);
      if(loadData) {
        pic->loadDataWithoutScaling(flags);
        assert(nFeatures==pic->mat.channels());
      }
      pictures.push_back(pic);
    }
  }
}

OpenCVUnlabeledDataSet::OpenCVUnlabeledDataSet
(std::string classesListFile, std::string dataDirectory, std::string wildcard,
 int backgroundCol,
 bool loadData, int flags) {
  name=dataDirectory;
  header="image";
  type=UNLABELEDBATCH;
  if (flags==0) nFeatures=1;
  if (flags==1) nFeatures=3;
  {
    std::ifstream f(classesListFile.c_str());
    std::string cl;
    int ctr=0;
    while (f >> cl) {
      classes[cl]=ctr++;
      header+=","+cl;
    }
  }
  nClasses=classes.size();
  for (auto &file : globVector(dataDirectory+"/"+wildcard)) {
    OpenCVPicture* pic = new OpenCVPicture(file,backgroundCol,0);
    if(loadData) {
      pic->loadDataWithoutScaling(flags);
      assert(nFeatures==pic->mat.channels());
    }
    pictures.push_back(pic);
  }
}
