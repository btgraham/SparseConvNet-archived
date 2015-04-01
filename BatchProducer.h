#pragma once

class SpatiallySparseCNN;

class BatchProducer {
public:
  SpatiallySparseCNN& cnn;
  int batchCounter;
  boost::thread_group workers;
  SpatialDataset& dataset;
  vector<SpatiallySparseBatch*> v;
  int batchSize;
  int nThreads;
  int spatialSize;
  vector<int> permutation;
  SpatiallySparseBatch* nextBatch();
  void batchProducerThread(int nThread);
  BatchProducer (SpatiallySparseCNN& cnn, SpatialDataset &dataset, int spatialSize, int batchSize, int nThreads);
  ~BatchProducer();
};
