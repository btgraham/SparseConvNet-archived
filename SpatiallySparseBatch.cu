#include "SpatiallySparseBatch.h"
#include "utilities.h"

SpatiallySparseBatch::SpatiallySparseBatch() {
}
void SpatiallySparseBatch::reset() {
  batchSize=0;
  sampleNumbers.resize(0);
  for (int i=0;i<interfaces.size();++i)
    interfaces[i].reset();
  labels.resize(0);
  predictions.resize(0);
  probabilities.resize(0);
  negativeLogLikelihood=0;
  mistakes=0;
  inputSub.reset();
}
