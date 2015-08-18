#include "SpatiallySparseDatasetKagglePlankton.h"
#include<iostream>
#include<fstream>
#include<string>
#include "utilities.h"

KagglePlanktonLabeledDataSet::KagglePlanktonLabeledDataSet
(std::string classesListFile, std::string dataDirectory, batchType type_, int backgroundCol) {
  name=dataDirectory;
  type=type_;
  {
    std::ifstream f(classesListFile.c_str());
    std::string cl;
    int ctr=0;
    while (f >> cl)
      classes[cl]=ctr++;
  }
  nClasses=classes.size();
  for (auto &kv : classes) {
    for (auto &file : globVector(dataDirectory+kv.first+"/*.jpg")) {
      OpenCVPicture* pic = new OpenCVPicture(file,backgroundCol,kv.second);
      pic->loadDataWithoutScaling(-1);
      nFeatures=pic->mat.channels();
      pic->scale=powf(powf(pic->mat.rows,2)+powf(pic->mat.cols,2),0.5);
      pictures.push_back(pic);
    }
  }
}

KagglePlanktonUnlabeledDataSet::KagglePlanktonUnlabeledDataSet
(std::string classesListFile, std::string dataDirectory, int backgroundCol) {
  name=dataDirectory;
  header="image";
  type=UNLABELEDBATCH;
  {
    std::ifstream f(classesListFile.c_str());
    std::string cl;
    int ctr=0;
    while (f >> cl) {
      classes[cl]=ctr++;
      header=header+","+cl;
    }
  }
  nClasses=classes.size();
  for (auto &file : globVector(std::string(dataDirectory+"*.jpg"))) {
    OpenCVPicture* pic = new OpenCVPicture(file,backgroundCol,0);
    pic->loadDataWithoutScaling(-1);
    nFeatures=pic->mat.channels();
    pic->scale=powf(powf(pic->mat.rows,2)+powf(pic->mat.cols,2),0.5);
    pictures.push_back(pic);
  }
}
