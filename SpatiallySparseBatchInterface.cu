#include <iostream>
#include "SpatiallySparseBatchInterface.h"

SpatiallySparseBatchInterface::SpatiallySparseBatchInterface() :
  featuresPresent(false,0) {
}
void SpatiallySparseBatchInterface::summary() {
  std::cout << "---------------------------------------------------\n";
  std::cout << "nFeatures" << nFeatures << std::endl;
  std::cout << "featuresPresent.size()" << featuresPresent.size() <<std::endl;
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
  nSpatialSites=0;
#ifdef USE_GOOGLE_SPARSEHASH
  grids.resize(0);
  // for (auto &grid : grids) {
  //   grid.backgroundCol=-1;
  //   grid.mp.clear_no_resize();
  // }
#else
  grids.resize(0);
#endif
  rules.resize(0);
  backpropErrors=false;
}
