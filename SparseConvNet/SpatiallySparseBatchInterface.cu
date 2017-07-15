#include <iostream>
#include "SpatiallySparseBatchInterface.h"

SpatiallySparseBatchSubInterface::SpatiallySparseBatchSubInterface() {
  reset();
}

void SpatiallySparseBatchSubInterface::reset() {
  features.resize(0);
  dfeatures.resize(0);
  poolingChoices.resize(0);
}

SpatiallySparseBatchInterface::SpatiallySparseBatchInterface(
    SpatiallySparseBatchSubInterface *s)
    : sub(s), rules(false, 0), featuresPresent(false, 0) {
  reset();
}
void SpatiallySparseBatchInterface::summary() {
  std::cout << "---------------------------------------------------\n";
  std::cout << "nFeatures" << nFeatures << std::endl;
  std::cout << "featuresPresent.size()" << featuresPresent.size() << std::endl;
  std::cout << "spatialSize" << spatialSize << std::endl;
  std::cout << "nSpatialSites" << nSpatialSites << std::endl;
  std::cout << "sub->features.size()" << sub->features.size() << std::endl;
  std::cout << "sub->dfeatures.size()" << sub->dfeatures.size() << std::endl;
  std::cout << "grids.size()" << grids.size() << std::endl;
  std::cout << "grids[0].mp.size()" << grids[0].mp.size() << std::endl;
  std::cout << "---------------------------------------------------\n";
}
void SpatiallySparseBatchInterface::reset() {
  featuresPresent.resize(0);
  featuresPresent.copyToCPU();
  nSpatialSites = 0;
  grids.resize(0);
  rules.resize(0);
  rules.copyToCPU();
  backpropErrors = false;
}
