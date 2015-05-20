#include "SpatiallySparseDataset.h"
#include <algorithm>
#include <iostream>
#include <chrono>

void SpatiallySparseDataset::summary() {
  std::cout << "Name:           " << name << std::endl;
  std::cout << "nPictures:      " << pictures.size() << std::endl;
  std::cout << "nClasses:       " << nClasses << std::endl;
  std::cout << "nFeatures:      " << nFeatures << std::endl;
  std::cout << "Type:           " << batchTypeNames[type]<<std::endl;
}
SpatiallySparseDataset SpatiallySparseDataset::extractValidationSet(float p) {
  SpatiallySparseDataset val;
  val.name=name+std::string(" Validation set");
  name=name+std::string(" minus Validation set");
  val.nClasses=nClasses;
  val.nFeatures=nFeatures;
  val.type=TESTBATCH;
  shuffle();
  int size=pictures.size()*p;
  for (;size>0;size--) {
    val.pictures.push_back(pictures.back());
    pictures.pop_back();
  }
  return val;
}
void SpatiallySparseDataset::subsetOfClasses(std::vector<int> activeClasses) {
  nClasses=activeClasses.size();
  std::vector<Picture*> p=pictures;
  pictures.clear();
  for (int i=0;i<p.size();++i) {
    std::vector<int>::iterator it;
    it = find (activeClasses.begin(), activeClasses.end(), p[i]->label);
    if (it != activeClasses.end()) {
      p[i]->label=it-activeClasses.begin();
      pictures.push_back(p[i]);
      //std::cout << pictures.size() << " " << p[i]->identify() << std::endl;
    } else
      delete p[i];
  }
}
SpatiallySparseDataset SpatiallySparseDataset::subset(int n) {
  SpatiallySparseDataset subset(*this);
  subset.shuffle();
  subset.pictures.resize(n);
  return subset;
}
void SpatiallySparseDataset::shuffle() {
  auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::mt19937 gen(seed);
  std::shuffle(pictures.begin(), pictures.end(), gen);
}
void SpatiallySparseDataset::repeatSamples(int reps) {
  int s=pictures.size();
  for (int i=1; i<reps; ++i)
    for (int j=0; j<s; ++j)
      pictures.push_back(pictures[j]);
}
