// don't delete the returned threads when done with them; ownership is with
// cnn.batchPool

#define MULTITHREAD_BATCH_PRODUCTION

#include "BatchProducer.h"
#include "SpatiallySparseBatch.h"
#include "SpatiallySparseBatchInterface.h"
#include "utilities.h"
#include <algorithm>
#include <functional>
#include <mutex>
#include <chrono>
#include <cassert>

BatchProducer::BatchProducer(SparseConvNetCUDA &cnn,
                             SpatiallySparseDataset &dataset, int spatialSize,
                             int batchSize)
    : cnn(cnn), batchCounter(-1), dataset(dataset), spatialSize(spatialSize),
      batchSize(batchSize) {
  assert(batchSize > 0);
  nBatches = (dataset.pictures.size() + batchSize - 1) / batchSize;
  permutation = range(dataset.pictures.size());
  if (dataset.type == TRAINBATCH) {
    RNG rng;
    rng.vectorShuffle(permutation);
  }
  while (cnn.batchPool[0].interfaces.size() <= cnn.layers.size()) {
    cnn.sharedSubInterfaces.push_back(new SpatiallySparseBatchSubInterface());
    for (int c = 0; c < cnn.nBatchProducerThreads; c++) {
      cnn.batchPool[c].interfaces.emplace_back(cnn.sharedSubInterfaces.back());
    }
  }
#ifdef MULTITHREAD_BATCH_PRODUCTION
  for (int nThread = 0; nThread < cnn.nBatchProducerThreads; ++nThread)
    workers.emplace_back(&BatchProducer::batchProducerThread, this, nThread);
#endif
}

void BatchProducer::preprocessBatch(int c, int cc, RNG &rng) {
  cnn.batchPool[cc].reset();
  cnn.batchPool[cc].type = dataset.type;
  cnn.batchPool[cc].interfaces[0].nFeatures = dataset.nFeatures;
  cnn.batchPool[cc].interfaces[0].spatialSize = spatialSize;
  cnn.batchPool[cc].interfaces[0].featuresPresent.hVector() =
      range(dataset.nFeatures);
  for (int i = c * batchSize;
       i < min((c + 1) * batchSize, (int)(dataset.pictures.size())); i++) {
    Picture *pic = dataset.pictures[permutation[i]]->distort(rng, dataset.type);
    cnn.batchPool[cc].sampleNumbers.push_back(permutation[i]);
    cnn.batchPool[cc].batchSize++;
    cnn.batchPool[cc].interfaces[0].grids.push_back(SparseGrid());
    cnn.batchPool[cc].labels.hVector().push_back(pic->label);
    pic->codifyInputData(
        cnn.batchPool[cc].interfaces[0].grids.back(),
        cnn.batchPool[cc].interfaces[0].sub->features.hVector(),
        cnn.batchPool[cc].interfaces[0].nSpatialSites,
        cnn.batchPool[cc].interfaces[0].spatialSize);
    if (pic != dataset.pictures[permutation[i]])
      delete pic;
  }
  assert(cnn.batchPool[cc].interfaces[0].sub->features.size() ==
         cnn.batchPool[cc].interfaces[0].nFeatures *
             cnn.batchPool[cc].interfaces[0].nSpatialSites);
  if (cnn.inputNormalizingConstants.size() > 0) {
    std::vector<float> &features =
        cnn.batchPool[cc].interfaces[0].sub->features.hVector();
    for (int i = 0; i < features.size(); ++i)
      features[i] *= cnn.inputNormalizingConstants
                         [i % (cnn.batchPool[cc].interfaces[0].nFeatures)];
  }
  for (int i = 0; i < cnn.layers.size(); i++)
    cnn.layers[i]->preprocess(cnn.batchPool[cc],
                              cnn.batchPool[cc].interfaces[i],
                              cnn.batchPool[cc].interfaces[i + 1]);
  // Shifted to line 126 !!!!!!
  // cnn.batchPool[cc].interfaces[0].sub->features.copyToGPUAsync(
  //     cnn.batchMemStreams[cc]);
  // cnn.batchPool[cc].labels.copyToGPUAsync(cnn.batchMemStreams[cc]);
  // for (int i = 0; i <= cnn.layers.size(); ++i) {
  //   cnn.batchPool[cc].interfaces[i].featuresPresent.copyToGPUAsync(
  //       cnn.batchMemStreams[cc]);
  //   cnn.batchPool[cc].interfaces[i].rules.copyToGPUAsync(
  //       cnn.batchMemStreams[cc]);
  // }
  // cudaStreamSynchronize(cnn.batchMemStreams[cc].stream);
}

#ifdef MULTITHREAD_BATCH_PRODUCTION
void BatchProducer::batchProducerThread(int nThread) {
  cudaSetDevice(cnn.deviceID);
  // cudaSafeCall(cudaStreamDestroy(cnn.batchMemStreams[nThread].stream));
  // cudaSafeCall(cudaStreamCreate(&cnn.batchMemStreams[nThread].stream));
  RNG rng;
  for (int c = nThread; c < nBatches; c += cnn.nBatchProducerThreads) {
    cnn.batchLock[nThread].lock();
    while (cnn.batchPool[nThread].batchSize >
           0) { // Don't overwrite unused batches
      cnn.batchLock[nThread].unlock();
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      cnn.batchLock[nThread].lock();
    }
    preprocessBatch(c, nThread, rng);
    cnn.batchLock[nThread].unlock();
  }
}

SpatiallySparseBatch *BatchProducer::nextBatch() {
  if (batchCounter >= 0) {
    int cc = batchCounter % cnn.nBatchProducerThreads;
    cnn.batchLock[cc].unlock();
    cnn.batchPool[cc].batchSize = 0;
  }
  batchCounter++;

  if (batchCounter < nBatches) {
    int cc = batchCounter % cnn.nBatchProducerThreads;
    for (bool ready = false; not ready;) {
      bool accessible = cnn.batchLock[cc].try_lock();
      if (accessible)
        if (cnn.batchPool[cc].batchSize == 0)
          cnn.batchLock[cc].unlock();
        else
          ready = true;
    }
    /////////////////////////////////////////////////////////
    cnn.batchPool[cc].interfaces[0].sub->features.copyToGPUAsync(cnn.memStream);
    cnn.batchPool[cc].labels.copyToGPUAsync(cnn.memStream);
    for (int i = 0; i <= cnn.layers.size(); ++i) {
      cnn.batchPool[cc].interfaces[i].featuresPresent.copyToGPUAsync(
          cnn.memStream);
      cnn.batchPool[cc].interfaces[i].rules.copyToGPUAsync(cnn.memStream);
    }
    cudaStreamSynchronize(cnn.memStream.stream);
    /////////////////////////////////////////////////////////
    return &cnn.batchPool[cc];
  } else {
    for (int i = 0; i < cnn.nBatchProducerThreads; i++)
      workers[i].join();
    return NULL;
  }
}

#else

void BatchProducer::batchProducerThread(int nThread) {}

SpatiallySparseBatch *BatchProducer::nextBatch() {
  if (batchCounter >= 0) {
    int cc = batchCounter % cnn.nBatchProducerThreads;
    cnn.batchPool[cc].batchSize = 0;
  }
  batchCounter++;
  if (batchCounter == nBatches) {
    return NULL;
  } else {
    RNG rng;
    int cc = batchCounter % cnn.nBatchProducerThreads;
    ;
    preprocessBatch(batchCounter, cc, rng);
    /////////////////////////////////////////////////////////
    cnn.batchPool[cc].interfaces[0].sub->features.copyToGPUAsync(cnn.memStream);
    cnn.batchPool[cc].labels.copyToGPUAsync(cnn.memStream);
    for (int i = 0; i <= cnn.layers.size(); ++i) {
      cnn.batchPool[cc].interfaces[i].featuresPresent.copyToGPUAsync(
          cnn.memStream);
      cnn.batchPool[cc].interfaces[i].rules.copyToGPUAsync(cnn.memStream);
    }
    cudaStreamSynchronize(cnn.memStream.stream);
    /////////////////////////////////////////////////////////
    return &cnn.batchPool[cc];
  }
}
#endif

BatchProducer::~BatchProducer() {
  if (batchCounter < nBatches) {
    SpatiallySparseBatch *batch = nextBatch();
    while (batch) {
      batch = nextBatch();
    }
  }
}
