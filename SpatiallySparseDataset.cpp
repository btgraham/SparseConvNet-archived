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
  std::mt19937 gen(123);
  std::shuffle(pictures.begin(), pictures.end(), gen);
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

//Assume there are at least n of each class in the dataset
SpatiallySparseDataset SpatiallySparseDataset::balancedSubset(int n) {
  SpatiallySparseDataset bs;
  bs.name=name+std::string(" subset");
  bs.nFeatures=nFeatures;
  bs.nClasses=nClasses;
  bs.type=type;
  shuffle();
  auto permutation=rng.permutation(pictures.size());
  std::vector<int> count(nClasses);
  int classesDone=0;
  for (int i=0;i<pictures.size() and classesDone<nClasses;i++) {
    auto pic=pictures[permutation[i]];
    if (count[pic->label]++ < n)
      bs.pictures.push_back(pic);
    if (count[pic->label]==n)
      classesDone++;
  }
  return bs;
}

SpatiallySparseDataset SpatiallySparseDataset::subset(int n) {
  SpatiallySparseDataset subset;
  subset.name=name+std::string(" subset");
  subset.nFeatures=nFeatures;
  subset.nClasses=nClasses;
  subset.type=type;
  auto pick=rng.NchooseM(pictures.size(),n);
  for (auto i: pick) {
    subset.pictures.push_back(pictures[i]);
  }
  return subset;
}



void SpatiallySparseDataset::shuffle() {
  std::shuffle( pictures.begin(), pictures.end(), rng.gen);
}
void SpatiallySparseDataset::repeatSamples(int reps) {
  int s=pictures.size();
  for (int i=1; i<reps; ++i)
    for (int j=0; j<s; ++j)
      pictures.push_back(pictures[j]);
}
