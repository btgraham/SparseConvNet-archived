#include "SpatiallySparseCNN.h"

void SpatiallySparseCNN::addLearntLayer(int nFeatures,
                                        ActivationFunction activationFn,
                                        float dropout,
                                        float alpha) {
  if (activationFn!=SOFTMAX)
    nFeatures=max(KERNELBLOCKSIZE,kernelBlockSizeRound(nFeatures));
  if (dropout>0)
    dropout=1-kernelBlockSizeRound(nFeatures*(1-dropout))*1.0f/nFeatures;
  cnn.push_back(new NetworkInNetworkLayer(nOutputFeatures, nFeatures, dropout, activationFn, alpha));
  nOutputFeatures=nFeatures;
}
void SpatiallySparseCNN::addNetworkInNetworkLayer(int nFeatures,
                                                  ActivationFunction activationFn,
                                                  float dropout) {
  addLearntLayer(nFeatures, activationFn, dropout, 1.0f);
}
void SpatiallySparseCNN::addConvolutionalLayer(int nFeatures,
                                               int filterSize,
                                               int filterStride,
                                               ActivationFunction activationFn,
                                               float dropout,
                                               float poolingToFollow) {
  if (filterSize>1) {
    cnn.push_back(new ConvolutionalLayer(filterSize, filterStride, nOutputFeatures));
    nOutputFeatures*=filterSize*filterSize;
  }
  addLearntLayer(nFeatures,activationFn,dropout,powf(filterSize*1.0/filterStride/poolingToFollow,2));
}
void SpatiallySparseCNN::addXConvLayer(int nFeatures,
                                       int filterSize,
                                       int filterStride,
                                       ActivationFunction activationFn,
                                       float dropout,
                                       float poolingToFollow) {
  assert(filterSize%2==1 and filterSize>1);
  cnn.push_back(new XConvLayer(filterSize, filterStride, nOutputFeatures));
  nOutputFeatures*=(2*filterSize-1);
  addLearntLayer(nFeatures,activationFn,dropout,filterSize*2.0/filterStride/filterStride/poolingToFollow/poolingToFollow);
}
void SpatiallySparseCNN::addX4ConvLayer(int nFeatures,
                                        ActivationFunction activationFn) {
  cnn.push_back(new X4ConvLayer(nOutputFeatures));
  nOutputFeatures*=4;
  addLearntLayer(nFeatures,activationFn);
}
void SpatiallySparseCNN::addYConvLayer(int nFeatures,
                                       int filterStride,
                                       ActivationFunction activationFn,
                                       float dropout,
                                       float poolingToFollow) {
  cnn.push_back(new YConvLayer(filterStride, nOutputFeatures));
  nOutputFeatures*=9;
  addLearntLayer(nFeatures,activationFn,dropout,9.0/filterStride/filterStride/poolingToFollow/poolingToFollow);
}
void SpatiallySparseCNN::addLeNetLayerMP(int nFeatures, int filterSize, int filterStride, int poolSize, int poolStride, ActivationFunction activationFn, float dropout) {
  addConvolutionalLayer(nFeatures,filterSize,filterStride,activationFn,dropout,poolSize);
  if (poolSize>1)
    cnn.push_back(new MaxPoolingLayer(poolSize, poolStride));
}
void SpatiallySparseCNN::addLeNetLayerAP(int nFeatures, int filterSize, int filterStride, int poolSize, int poolStride, ActivationFunction activationFn, float dropout) {
  addConvolutionalLayer(nFeatures,filterSize,filterStride,activationFn,dropout,poolSize);
  if (poolSize>1)
    cnn.push_back(new AveragePoolingLayer(poolSize, poolStride));
}
void SpatiallySparseCNN::addLeNetLayerROFMP(int nFeatures, int filterSize, int filterStride, float fmpShrink, ActivationFunction activationFn, float dropout) {
  addConvolutionalLayer(nFeatures,filterSize,filterStride,activationFn,dropout,fmpShrink);
  if (fmpShrink>1) {
    cnn.push_back(new RandomOverlappingFractionalMaxPoolingLayer(fmpShrink));
  }
}
void SpatiallySparseCNN::addLeNetLayerPOFMP(int nFeatures, int filterSize, int filterStride, float fmpShrink, ActivationFunction activationFn, float dropout) {
  addConvolutionalLayer(nFeatures,filterSize,filterStride,activationFn,dropout,fmpShrink);
  if (fmpShrink>1)
    cnn.push_back(new PseudorandomOverlappingFractionalMaxPoolingLayer(fmpShrink));
}
void SpatiallySparseCNN::addLeNetLayerJOFMP(int nFeatures, int filterSize, int filterStride, float fmpShrink, int poolSize, ActivationFunction activationFn, float dropout) {
  addConvolutionalLayer(nFeatures,filterSize,filterStride,activationFn,dropout,fmpShrink);
  if (fmpShrink>1)
    cnn.push_back(new JitteryOverlappingFractionalMaxPoolingLayer(fmpShrink, poolSize));
}
void SpatiallySparseCNN::addSoftmaxLayer() {
  addLearntLayer(nClasses, SOFTMAX,0.0f,10000);
  inputSpatialSize=1;
  for (int i=cnn.size()-1;i>=0;i--) {
    inputSpatialSize=cnn[i]->calculateInputSpatialSize(inputSpatialSize);
  }
  cout <<"Spatially sparse CNN: input size " << inputSpatialSize <<"x"<<inputSpatialSize<<endl;
}
void SpatiallySparseCNN::addIndexLearnerLayer() {
  cnn.push_back(new IndexLearnerLayer(nOutputFeatures, nClasses));
  nOutputFeatures=nClasses; // "nClasses"=trainingSet.pictures.size()
  inputSpatialSize=1;
  for (int i=cnn.size()-1;i>=0;i--) {
    inputSpatialSize=cnn[i]->calculateInputSpatialSize(inputSpatialSize);
  }
  cout <<"Spatially sparse CNN: input size " << inputSpatialSize <<"x"<<inputSpatialSize<<endl;
}
void SpatiallySparseCNN::processBatch(SpatiallySparseBatch& batch, float learningRate, ofstream& f, ofstream& g) {
  for (int i=0;i<cnn.size();i++) {
    cnn[i]->forwards(batch.interfaces[i],batch.interfaces[i+1]);
    if (batch.interfaces[0].type==RESCALEBATCH and i<cnn.size()-1)
      cnn[i]->scaleWeights(batch.interfaces[i],batch.interfaces[i+1]);
  }
  SoftmaxClassifier(batch.interfaces.back(),batch,nTop);
  if (batch.interfaces[0].type==TRAINBATCH)
    for (int i=cnn.size()-1;i>=0;i--)
      cnn[i]->backwards(batch.interfaces[i],batch.interfaces[i+1],learningRate);
  if (f)
    for (int j=0;j<batch.predictions.size();j++) {
      for (int k=0;k<batch.predictions[j].size();k++) {
        if (k>0) f << " ";
        f << batch.predictions[j][k];
      }
      f << endl;
    }
  if (g)
    for (int j=0;j<batch.predictions.size();j++) {
      for (int k=0;k<batch.probabilities[j].size();k++) {
        if (k>0) g << " ";
        g << batch.probabilities[j][k];
      }
      g << endl;
    }
}
float SpatiallySparseCNN::processDataset(SpatialDataset &dataset, int batchSize, float learningRate) {
  float errorRate=0, nll=0;
  ofstream f,g;
  BatchProducer bp(*this, dataset,inputSpatialSize,batchSize,4);
  if (dataset.type==UNLABELLEDBATCH) {
    f.open("unlabelledData.predictions");
    g.open("unlabelledData.probabilities");
  }
  while(SpatiallySparseBatch* batch=bp.nextBatch()) {
    processBatch(*batch,learningRate,f,g);
    errorRate+=batch->mistakes*1.0/dataset.pictures.size();
    nll+=batch->negativeLogLikelihood*1.0/dataset.pictures.size();
    delete batch;
  }
  cout << dataset.name
       << " Mistakes: "
       << 100*errorRate
       << "% NLL "
       << nll
       <<endl;
  return errorRate;
}
void SpatiallySparseCNN::processDatasetRepeatTest(SpatialDataset &dataset, int batchSize, int nReps, string predictionsFilename,string header) {
  vector<vector<int> >   votes(dataset.pictures.size());
  vector<vector<float> > probs(dataset.pictures.size());
  for (int i=0;i<dataset.pictures.size();++i) {
    votes[i].resize(dataset.nClasses);
    probs[i].resize(dataset.nClasses);
  }
  ofstream f,g;
  for (int rep=1;rep<=nReps;++rep) {
    BatchProducer bp(*this, dataset,inputSpatialSize,batchSize,4);
    while(SpatiallySparseBatch* batch=bp.nextBatch()) {
      processBatch(*batch,0,f,g);
      for (int i=0;i<batch->interfaces[0].batchSize;++i) {
        int ii=batch->sampleNumbers[i];
        votes[ii][batch->predictions[i][0]]++;
        for (int j=0;j<dataset.nClasses;++j)
          probs[ii][j]+=batch->probabilities[i][j];
      }
      delete batch;
    }
    float errors=dataset.pictures.size(),nll=0;
    for (int i=0;i<dataset.pictures.size();++i) {
      vector<int> predictions=vectorTopIndices(probs[i],nTop);
      for (int j=0;j<nTop;j++)
        if (predictions[j]==dataset.pictures[i]->label)
          errors--;
      nll-=log(max(probs[i][dataset.pictures[i]->label]/rep,1.0e-15));
    }
    if (!predictionsFilename.empty()) {
      cout << predictionsFilename << endl;
      f.open(predictionsFilename.c_str());
      f<<header;
      for (int i=0;i<dataset.pictures.size();++i) {
        f << dataset.pictures[i]->identify();
        for (int j=0;j<dataset.nClasses;++j)
          f <<"," << probs[i][j]/rep;
        f <<endl;
      }
      f.close();

    }
    cout << dataset.name
         << " rep " << rep <<"/"<<nReps
         << " Mistakes: " << 100*errors/dataset.pictures.size()
         << "% NLL " << nll/dataset.pictures.size()
         <<endl;
    // // if (dataset.type==UNLABELLEDBATCH and rep==nReps) {
    // //   ofstream f;f.open("unlabelledData.predictions");
    // //   ofstream g;g.open("unlabelledData.probabilities");
    // // }
  }

}
void SpatiallySparseCNN::loadWeights(string baseName, int epoch, int firstNlayers) {
  string filename=string(baseName)+string("_epoch-")+boost::lexical_cast<string>(epoch)+string(".cnn");
  ifstream f;
  f.open(filename.c_str(),ios::out | ios::binary);
  if (f) {
    cout << "Loading network parameters from " << filename << endl;
  } else {
    cout <<"Cannot find " << filename << endl;
    exit(EXIT_FAILURE);
  }
  for (int i=0;i<min((int)cnn.size(),firstNlayers);i++)
    cnn[i]->loadWeightsFromStream(f);
  f.close();
}
void SpatiallySparseCNN::saveWeights(string baseName, int epoch)  {
  string filename=string(baseName)+string("_epoch-")+boost::lexical_cast<string>(epoch)+string(".cnn");
  ofstream f;
  f.open(filename.c_str(),ios::binary);
  if (f) {
    for (int i=0;i<cnn.size();i++)
      cnn[i]->putWeightsToStream(f);
    f.close();
  } else {
    cout <<"Cannot write " << filename << endl;
    exit(EXIT_FAILURE);
  }
}
void SpatiallySparseCNN::processIndexLearnerBatch(SpatiallySparseBatch& batch, float learningRate, ofstream& f) {
  int n=cnn.size();
  for (int i=0;i<n-1;i++)
    cnn[i]->forwards(batch.interfaces[i],batch.interfaces[i+1]);  //Stop 1 early unless it is a training batch
  if (f.is_open()) {
    assert(batch.interfaces.back().nFeatures==batch.interfaces.back().featuresPresent.size());
    for (int i=0;i<batch.interfaces[0].batchSize;i++) {
      f << batch.sampleNumbers[i] << " " << batch.labels.hVector()[i];
      for (int j=0;j<batch.interfaces.back().nFeatures;j++)
        f << " " << batch.interfaces.back().features.hVector()[i*batch.interfaces.back().nFeatures+j];
      f << endl;
    }
  }
  if (batch.interfaces[0].type==TRAINBATCH) {
    dynamic_cast<IndexLearnerLayer*>(cnn[n-1])->indexLearnerIndices=batch.sampleNumbers;
    cnn[n-1]->forwards(batch.interfaces[n-1],batch.interfaces[n]);
    IndexLearner(batch.interfaces[n],batch,nTop);
    for (int i=n-1;i>=0;i--)
      cnn[i]->backwards(batch.interfaces[i],batch.interfaces[i+1],learningRate);
  }
}
float SpatiallySparseCNN::processIndexLearnerDataset(SpatialDataset &dataset, int batchSize, float learningRate) {
  float errorRate=0, nll=0;
  ofstream f;
  BatchProducer bp(*this, dataset,inputSpatialSize,batchSize,4);
  if (dataset.type!=TRAINBATCH) {
    string filename=dataset.name+".features";
    f.open(filename.c_str());
  }
  while(SpatiallySparseBatch* batch=bp.nextBatch()) {
    processIndexLearnerBatch(*batch,learningRate,f);
    errorRate+=batch->mistakes*1.0/dataset.pictures.size();
    nll+=batch->negativeLogLikelihood*1.0/dataset.pictures.size();
    delete batch;
  }
  if (dataset.type==TRAINBATCH)
    cout << dataset.name
         << " Mistakes: "
         << 100*errorRate
         << "% NLL "
         << nll
         <<endl;
  return errorRate;
}
void SpatiallySparseCNN::processBatchDumpTopLevelFeaturess(SpatiallySparseBatch& batch, ofstream& f) {
  batch.interfaces.resize(cnn.size());
  for (int i=0;i<cnn.size()-1;i++) {
    cnn[i]->forwards(batch.interfaces[i],batch.interfaces[i+1]);
  }
  assert(batch.interfaces.back().nFeatures==batch.interfaces.back().featuresPresent.size());
  for (int i=0;i<batch.interfaces[0].batchSize;i++) {
    f << batch.sampleNumbers[i] << " " << batch.labels.hVector()[i];
    for (int j=0;j<batch.interfaces.back().nFeatures;j++)
      f << " " << batch.interfaces.back().features.hVector()[i*batch.interfaces.back().nFeatures+j];
    f << endl;
  }
}
void SpatiallySparseCNN::processDatasetDumpTopLevelFeatures(SpatialDataset &dataset, int batchSize) {
  ofstream f;
  assert(dataset.type!=TRAINBATCH);
  BatchProducer bp(*this, dataset,inputSpatialSize,batchSize,4);
  string filename=dataset.name+".features";
  f.open(filename.c_str());
  while(SpatiallySparseBatch* batch=bp.nextBatch()) {
    processBatchDumpTopLevelFeaturess(*batch,f);
    delete batch;
  }
}
