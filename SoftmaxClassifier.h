////////////////////////////////////////////////////////////////////////////////////////////////
//Calculate softmaxProbability(i) - indicator(i=label)
// for i=0,1,...N-1 with N the number of character classes.
__global__ void dDerivativeOfCostWRTpreSoftmaxTopLevelWeights
(int batchSize, float* topDelta, float* topGrid, int* labels, int N) {
  for (int k=0;k<batchSize;k++) {
    for(int i=threadIdx.x;i<N;i+=NTHREADS) {
      topDelta[k*N+i]=topGrid[k*N+i]-(i==labels[k]);
    }
  }
}

void SoftmaxClassifier(SpatiallySparseBatchInterface& input, SpatiallySparseBatch& batch, int nTop) {
  //Assume no dropout in the output layer! nClasses:=input.nFeatures.
  assert(input.batchSize==input.nSpatialSites);
  assert(input.nFeatures==input.featuresPresent.size());

  float* probs=&input.features.hVector()[0];
  for (int i=0;i<input.batchSize;++i)
    batch.probabilities.push_back(vector<float> (probs+i*input.nFeatures,probs+(i+1)*input.nFeatures));
  for (int i=0;i<input.batchSize;i++)
    batch.predictions.push_back(vectorTopIndices(batch.probabilities[i],nTop));

  if (input.type!=UNLABELLEDBATCH) {
    batch.mistakes+=input.batchSize;
    for (int i=0;i<input.batchSize;i++) {
      batch.negativeLogLikelihood-=log(max(batch.probabilities[i][batch.labels.hVector()[i]],1.0e-15));
      for (int j=0;j<nTop;j++) {
        if (batch.predictions[i][j]==batch.labels.hVector()[i]) {
          batch.mistakes--;
        }
      }
    }
  }
  //cout << batch.mistakes << " " << flush;
  //cout << (int)batch.negativeLogLikelihood << " " << flush;
  if (input.type==TRAINBATCH) {//Begin backprop. Top layer: d Cost / d SoftmaxInput
    input.dfeatures.resize(input.nSpatialSites*input.featuresPresent.size());
    dDerivativeOfCostWRTpreSoftmaxTopLevelWeights<<<1,NTHREADS>>>
      (input.batchSize, input.dfeatures.dPtr(), input.features.dPtr(),
       batch.labels.dPtr(), input.nFeatures);
  }
  cudaCheckError();
}
