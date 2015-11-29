#pragma once
#include <vector>
#include "SpatiallySparseBatchInterface.h"
#include "vectorCUDA.h"
#include "types.h"

class SpatiallySparseBatch {
  SpatiallySparseBatchSubInterface *inputSub;

public:
  SpatiallySparseBatch(SpatiallySparseBatchSubInterface *inputSub);
  batchType type;
  int batchSize;
  std::vector<int> sampleNumbers;
  std::vector<SpatiallySparseBatchInterface> interfaces;
  vectorCUDA<int> labels;
  std::vector<std::vector<int>> predictions;
  std::vector<std::vector<float>> probabilities;
  float negativeLogLikelihood;
  int mistakes;
  void reset();
};
