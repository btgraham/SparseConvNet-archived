#include "BatchProducer.h"
#define USE_THREADS_TO_PRODUCE_BATCHES
#ifdef USE_THREADS_TO_PRODUCE_BATCHES
SpatiallySparseBatch* BatchProducer::nextBatch() {
  if (batchCounter<v.size()) {
    while (v[batchCounter]==NULL)
      boost::this_thread::sleep(boost::posix_time::milliseconds(10));
    return v[batchCounter++];
  } else {
    workers.join_all();
    return NULL;
  }
}

void BatchProducer::batchProducerThread(int nThread) {
  RNG rng;
  for (int c=nThread;c<v.size();c+=nThreads) {
    while (c>batchCounter+25*nThreads)
      boost::this_thread::sleep(boost::posix_time::milliseconds(10));
    SpatiallySparseBatch* batch =
      new SpatiallySparseBatch(dataset.type, dataset.nFeatures, spatialSize,cnn.cnn.size()+1);
    for (int i=c*batchSize;i<min((c+1)*batchSize,(int)(dataset.pictures.size()));i++) {
      batch->sampleNumbers.push_back(permutation[i]);
      Picture* pic=dataset.pictures[permutation[i]]->distort(rng,dataset.type);
      pic->codifyInputData(*batch);
      if (pic!=dataset.pictures[permutation[i]])
        delete pic;
    }
    for (int i=0;i<cnn.cnn.size();i++)
      cnn.cnn[i]->preprocess(batch->interfaces[i],batch->interfaces[i+1]);
    assert(batch->interfaces[0].features.size()==batch->interfaces[0].nFeatures*batch->interfaces[0].nSpatialSites);
    v[c]=batch;
  }
}

BatchProducer::BatchProducer (SpatiallySparseCNN& cnn, SpatialDataset &dataset, int spatialSize, int batchSize, int nThreads) :
  cnn(cnn), batchCounter(0), dataset(dataset), spatialSize(spatialSize), batchSize(batchSize), nThreads(nThreads){
  v.resize((dataset.pictures.size()+batchSize-1)/batchSize,NULL);
  permutation=range(dataset.pictures.size());
  if (dataset.type==TRAINBATCH)
    random_shuffle ( permutation.begin(), permutation.end());
  for (int nThread=0; nThread<nThreads; nThread++)
    workers.add_thread(new boost::thread(boost::bind(&BatchProducer::batchProducerThread,this,nThread)));
}
BatchProducer::~BatchProducer() {
  SpatiallySparseBatch* batch=nextBatch();
  while(batch) {
    delete batch;
    batch=nextBatch();
  }
}
#else
SpatiallySparseBatch* BatchProducer::nextBatch() {
  RNG rng;
  if (batchCounter<v.size()) {
    SpatiallySparseBatch* batch =
      new SpatiallySparseBatch(dataset.type, dataset.nFeatures, spatialSize,cnn.cnn.size()+1);
    for (int i=batchCounter*batchSize;i<min((batchCounter+1)*batchSize,(int)(dataset.pictures.size()));i++) {
      batch->sampleNumbers.push_back(permutation[i]);
      Picture* pic=dataset.pictures[permutation[i]]->distort(rng,dataset.type);
      pic->codifyInputData(*batch);
      if (pic!=dataset.pictures[permutation[i]])
        delete pic;
    }
    assert(batch->interfaces[0].features.size()==batch->interfaces[0].nFeatures*batch->interfaces[0].nSpatialSites);
    for (int i=0; i<cnn.cnn.size();i++)
      cnn.cnn[i]->preprocess(batch->interfaces[i],batch->interfaces[i+1]);
    v[batchCounter]=batch;
    return v[batchCounter++];
  } else
    return NULL;
}

void BatchProducer::batchProducerThread(int nThread) {
}

BatchProducer::BatchProducer (SpatiallySparseCNN& cnn, SpatialDataset &dataset, int spatialSize, int batchSize, int nThreads) :
  cnn(cnn), batchCounter(0), dataset(dataset), spatialSize(spatialSize), batchSize(batchSize), nThreads(nThreads){
  v.resize((dataset.pictures.size()+batchSize-1)/batchSize,NULL);
  permutation=range(dataset.pictures.size());
  if (dataset.type==TRAINBATCH)
    random_shuffle ( permutation.begin(), permutation.end());
}
BatchProducer::~BatchProducer() {
  SpatiallySparseBatch* batch=nextBatch();
  while(batch) {
    delete batch;
    batch=nextBatch();
  }
}
#endif
