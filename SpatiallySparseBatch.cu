#include "SpatiallySparseBatch.h"
#include "utilities.h"

SpatiallySparseBatch::SpatiallySparseBatch(
    SpatiallySparseBatchSubInterface *inputSub) {
  interfaces.emplace_back(inputSub);
  reset();
}
void SpatiallySparseBatch::reset() {
  batchSize = 0;
  sampleNumbers.resize(0);
  for (int i = 0; i < interfaces.size(); ++i)
    interfaces[i].reset();
  interfaces[0].sub->reset();
  labels.resize(0);
  predictions.resize(0);
  probabilities.resize(0);
  negativeLogLikelihood = 0;
  mistakes = 0;
}
