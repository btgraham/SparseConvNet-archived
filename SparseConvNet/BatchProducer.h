#pragma once
#include <thread>
#include <vector>
#include "SpatiallySparseBatch.h"
#include "SpatiallySparseDataset.h"
#include "SparseConvNetCUDA.h"

class BatchProducer {
public:
  SparseConvNetCUDA &cnn;
  int batchCounter;
  int nBatches;
  std::vector<std::thread> workers;
  SpatiallySparseDataset &dataset;
  int batchSize;
  int spatialSize;
  std::vector<int> permutation;
  BatchProducer(SparseConvNetCUDA &cnn, SpatiallySparseDataset &dataset,
                int spatialSize, int batchSize);
  ~BatchProducer();
  SpatiallySparseBatch *nextBatch();
  void batchProducerThread(int nThread);
  void preprocessBatch(int c, int cc, RNG &rng);
};
