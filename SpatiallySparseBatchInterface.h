#pragma once
#include <vector>
#include "SparseGrid.h"
#include "vectorCUDA.h"

// This is a subset of the whole interface.
// It contains the larger vectors that can mostly stay on the GPU
class SpatiallySparseBatchSubInterface {
public:
  // In the input layer, this is configured during preprocessing
  vectorCUDA<float> features;
  // For the backwards/backpropagation pass
  vectorCUDA<float> dfeatures;
  vectorCUDA<int> poolingChoices;
  SpatiallySparseBatchSubInterface();
  void reset();
};

class SpatiallySparseBatchInterface {
public:
  SpatiallySparseBatchSubInterface *sub;
  int nFeatures; // Features per spatial location
  vectorCUDA<int>
      featuresPresent; // Not dropped out features per spatial location
  //                                          For dropout
  //                                          rng.NchooseM(nFeatures,featuresPresent.size());
  int nSpatialSites; // Total active spatial locations within the
  int spatialSize;   // spatialSize x spatialSize grid
  //                                          batchSize x spatialSize x
  //                                          spatialSize
  //                                          possible locations.
  bool backpropErrors; // Calculate dfeatures? (false until after the first NiN
                       // layer)
  std::vector<SparseGrid> grids; // batchSize vectors of maps storing info on
                                 // grids of size spatialSize x spatialSize
  //                                          Store locations of nSpatialSites
  //                                          in the
  //                                          spatialSize x spatialSize grids
  //                                          -1 entry corresponds to null
  //                                          vectors in needed
  // Below used internally for convolution/pooling operation:
  vectorCUDA<int> rules;
  SpatiallySparseBatchInterface(SpatiallySparseBatchSubInterface *s);
  void summary();
  void reset();
};
