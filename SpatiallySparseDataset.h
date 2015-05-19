#pragma once
#include "Picture.h"
#include <vector>
#include <string>
#include "types.h"
#include "SpatiallySparseDataset.h"


class SpatiallySparseDataset {
public:
  std::string name;
  std::vector<Picture*> pictures;
  int nFeatures;
  int nClasses;
  batchType type;
  void summary();
  void shuffle(); // For use before extracting a validation set (not for making gradient-descent stochastic)
  SpatiallySparseDataset extractValidationSet(float p=0.1);
  void subsetOfClasses(std::vector<int> activeClasses);
  SpatiallySparseDataset subset(int n);
  void repeatSamples(int reps); // Make dataset seem n times bigger (i.e. for small datasets to avoid having v. small training epochs)
};
