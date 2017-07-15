#include "SpatiallySparseDataset.h"
#include <algorithm>
#include <iostream>
#include <chrono>
#include <vector>

void SpatiallySparseDataset::summary() {
  std::cout << "Name:            " << name << std::endl;
  std::cout << "Type:            " << batchTypeNames[type] << std::endl;
  std::cout << "nFeatures:       " << nFeatures << std::endl;
  std::cout << "nPictures:       " << pictures.size() << std::endl;
  std::cout << "nClasses:        " << nClasses << std::endl;
  std::vector<int> count(nClasses);
  for (auto pic : pictures) {
    count[pic->label]++;
  }
  if (type != UNLABELEDBATCH) {
    std::cout << "nPictures/class:";
    for (auto i : count)
      std::cout << " " << i;
    std::cout << std::endl;
  }
}
SpatiallySparseDataset SpatiallySparseDataset::extractValidationSet(float p) {
  SpatiallySparseDataset val;
  val.name = name + std::string(" Validation set");
  name = name + std::string(" minus Validation set");
  val.nClasses = nClasses;
  val.nFeatures = nFeatures;
  val.type = TESTBATCH;
  std::mt19937 gen(123);
  std::shuffle(pictures.begin(), pictures.end(), gen);
  int size = pictures.size() * p;
  for (; size > 0; size--) {
    val.pictures.push_back(pictures.back());
    pictures.pop_back();
  }
  return val;
}
void SpatiallySparseDataset::subsetOfClasses(std::vector<int> activeClasses) {
  nClasses = activeClasses.size();
  std::vector<Picture *> p = pictures;
  pictures.clear();
  for (unsigned int i = 0; i < p.size(); ++i) {
    std::vector<int>::iterator it;
    it = find(activeClasses.begin(), activeClasses.end(), p[i]->label);
    if (it != activeClasses.end()) {
      p[i]->label = it - activeClasses.begin();
      pictures.push_back(p[i]);
      // std::cout << pictures.size() << " " << p[i]->identify() << std::endl;
    } else
      delete p[i];
  }
}

// Pick n samples from each class if possible
SpatiallySparseDataset SpatiallySparseDataset::balancedSample(int n) {
  SpatiallySparseDataset bs;
  bs.name = name + std::string(" sample");
  bs.nFeatures = nFeatures;
  bs.nClasses = nClasses;
  bs.type = type;
  std::vector<int> count(nClasses);
  int classesDone = 0;
  while (classesDone < nClasses) {
    auto permutation = rng.permutation(pictures.size());
    bool makingProgress = false;
    for (unsigned int i = 0; i < pictures.size() and classesDone < nClasses;
         i++) {
      auto pic = pictures[permutation[i]];
      if (count[pic->label]++ < n) {
        bs.pictures.push_back(pic);
        makingProgress = true;
      }
      if (count[pic->label] == n)
        classesDone++;
    }
    if (not makingProgress) {
      // std::cerr << "SpatiallySparseDataset::balancedSample(...) cannot find "
      //              "examples of all classes in " << name << std::endl;
      break;
    }
  }
  return bs;
}

SpatiallySparseDataset SpatiallySparseDataset::subset(int n) {
  SpatiallySparseDataset subset;
  subset.name = name + std::string(" subset");
  subset.nFeatures = nFeatures;
  subset.nClasses = nClasses;
  subset.type = type;
  auto pick = rng.NchooseM(pictures.size(), n);
  for (auto i : pick) {
    subset.pictures.push_back(pictures[i]);
  }
  return subset;
}

void SpatiallySparseDataset::shuffle() {
  std::shuffle(pictures.begin(), pictures.end(), rng.gen);
}
void SpatiallySparseDataset::repeatSamples(int reps) {
  std::cout << "Repeating each sample in " << name << " " << reps << " times"
            << std::endl;
  int s = pictures.size();
  for (int i = 1; i < reps; ++i)
    for (int j = 0; j < s; ++j)
      pictures.push_back(pictures[j]);
}

// http://stackoverflow.com/questions/612097/how-can-i-get-the-list-of-files-in-a-directory-using-c-or-c
std::vector<std::string> globVector(const std::string &pattern) {
  glob_t glob_result;
  glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
  std::vector<std::string> files;
  for (unsigned int i = 0; i < glob_result.gl_pathc; ++i) {
    files.push_back(std::string(glob_result.gl_pathv[i]));
  }
  globfree(&glob_result);
  std::sort(files.begin(), files.end());
  return files;
}

// Usage: std::vector<std::string> files = globVector("./*");
