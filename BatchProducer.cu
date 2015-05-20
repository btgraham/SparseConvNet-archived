#include "BatchProducer.h"
#include "SpatiallySparseBatch.h"
#include "utilities.h"
#include "cudaUtilities.h"
#include <functional>
#include <mutex>
#include <chrono>
#define NBATCHPRODUCERTHREADS 10

bool batchProducerBatchesInitialized; //intially false
std::mutex batchLock[NBATCHPRODUCERTHREADS];
SpatiallySparseBatch* batchPool[NBATCHPRODUCERTHREADS];
cudaMemStream* batchMemStreams[NBATCHPRODUCERTHREADS];

//don't delete the returned threads when done with them; ownership is with batchPool
SpatiallySparseBatch* BatchProducer::nextBatch() {
  if (batchCounter>=0) {
    batchPool[batchCounter%NBATCHPRODUCERTHREADS]->reset();
    batchLock[batchCounter%NBATCHPRODUCERTHREADS].unlock();
  }
  batchCounter++;

  if (batchCounter<nBatches) {
    int cc=batchCounter%NBATCHPRODUCERTHREADS;
    for(bool ready=false;!ready;) {
      ready=batchLock[cc].try_lock();
      if (ready and batchPool[cc]->batchSize==0) {   //not really ready
        batchLock[cc].unlock();
        ready=false;
      }
    }
    return batchPool[cc];
  } else {
    for (int i=0;i<NBATCHPRODUCERTHREADS;i++)
      workers[i].join();
    return NULL;
  }
}

void BatchProducer::batchProducerThread(int nThread) {
  cudaSetDevice(cnn.deviceID);
  RNG rng;
  for (int c=nThread;c<nBatches;c+=NBATCHPRODUCERTHREADS) {
    int cc=c%NBATCHPRODUCERTHREADS;
    batchLock[cc].lock();
    while(batchPool[cc]->batchSize>0) { //Don't overwrite unused batches
      batchLock[cc].unlock();
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      batchLock[cc].lock();
    }
    batchPool[cc]->type=dataset.type;
    batchPool[cc]->interfaces[0].nFeatures=dataset.nFeatures;
    batchPool[cc]->interfaces[0].spatialSize=spatialSize;
    batchPool[cc]->interfaces[0].featuresPresent.hVector()=range(dataset.nFeatures);
    for (int i=c*batchSize;i<min((c+1)*batchSize,(int)(dataset.pictures.size()));i++) {
      Picture* pic=dataset.pictures[permutation[i]]->distort(rng,dataset.type);
      batchPool[cc]->sampleNumbers.push_back(permutation[i]);
      batchPool[cc]->batchSize++;
      batchPool[cc]->interfaces[0].grids.push_back(SparseGrid());
      batchPool[cc]->labels.hVector().push_back(pic->label);
      pic->codifyInputData
        (batchPool[cc]->interfaces[0].grids.back(),
         batchPool[cc]->interfaces[0].sub->features.hVector(),
         batchPool[cc]->interfaces[0].nSpatialSites,
         batchPool[cc]->interfaces[0].spatialSize);
      if (pic!=dataset.pictures[permutation[i]])
        delete pic;
    }
    assert(batchPool[cc]->interfaces[0].sub->features.size()
           ==
           batchPool[cc]->interfaces[0].nFeatures*batchPool[cc]->interfaces[0].nSpatialSites);
    if (cnn.inputNormalizingConstants.size()>0) {
      std::vector<float> &features=batchPool[cc]->interfaces[0].sub->features.hVector();
      for (int i=0;i<features.size();++i)
        features[i]*=cnn.inputNormalizingConstants[i%(batchPool[cc]->interfaces[0].nFeatures)];
    }
    for (int i=0; i<cnn.layers.size();i++)
      cnn.layers[i]->preprocess(*batchPool[cc],batchPool[cc]->interfaces[i],batchPool[cc]->interfaces[i+1]);

    batchPool[cc]->interfaces[0].sub->features.copyToGPUAsync(*batchMemStreams[cc]);
    batchPool[cc]->labels.copyToGPUAsync(*batchMemStreams[cc]);
    for (int i=0;i<=cnn.layers.size();++i) {
      batchPool[cc]->interfaces[i].featuresPresent.copyToGPUAsync(*batchMemStreams[cc]);
      batchPool[cc]->interfaces[i].rules.copyToGPUAsync(*batchMemStreams[cc]);
    }

    batchLock[cc].unlock();
  }
}

BatchProducer::BatchProducer (SparseConvNetCUDA& cnn,
                              SpatiallySparseDataset &dataset,
                              int spatialSize, int batchSize) :
  cnn(cnn), batchCounter(-1),
  dataset(dataset),
  spatialSize(spatialSize), batchSize(batchSize) {
  if (!batchProducerBatchesInitialized) {
    for (int c=0;c<NBATCHPRODUCERTHREADS;c++) {
      //Set up streams and pinned memory for "copyToGPUAsync" operations
      batchMemStreams[c]=new cudaMemStream();
      //Set up shared memory
      batchPool[c]=new SpatiallySparseBatch();
      batchPool[c]->interfaces.resize(cnn.layers.size()+1);
      batchPool[c]->interfaces[0].sub = &batchPool[c]->inputSub; //unique for each batch
      for (int i=0;i<cnn.layers.size();++i)
        batchPool[c]->interfaces[i+1].sub=&cnn.layers[i]->sub; //shared between batches
      //Initialize batches
      batchPool[c]->reset();
    }
    batchProducerBatchesInitialized=true;
  }
  nBatches=(dataset.pictures.size()+batchSize-1)/batchSize;
  permutation=range(dataset.pictures.size());
  if (dataset.type==TRAINBATCH)
    random_shuffle ( permutation.begin(), permutation.end());
  for (int nThread=0; nThread<NBATCHPRODUCERTHREADS; nThread++)
    workers.emplace_back(std::thread(std::bind(&BatchProducer::batchProducerThread,this,nThread)));
}
BatchProducer::~BatchProducer() {
  if (batchCounter<nBatches) {
    SpatiallySparseBatch* batch=nextBatch();
    while(batch) {
      batch->reset();
      batch=nextBatch();
    }
  }
}


void batchProducerResourcesCleanup() {
  if (batchProducerBatchesInitialized) {
    for (int c=0;c<NBATCHPRODUCERTHREADS;c++) {
      delete batchPool[c];
      delete batchMemStreams[c];

    }
    batchProducerBatchesInitialized=false;
  }
}
