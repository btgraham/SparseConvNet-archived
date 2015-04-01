//See Ben Graham http://www2.warwick.ac.uk/fac/sci/statistics/staff/academic-research/graham/indexlearning.pdf
//and
//http://papers.nips.cc/paper/5548-discriminative-unsupervised-feature-learning-with-convolutional-neural-networks.pdf


class IndexLearnerLayer : public SpatiallySparseLayer {
private:
  RNG rng;
  vectorCUDA<float> W; //Weights
  vectorCUDA<float> MW; //momentum
  vectorCUDA<float> w; //shrunk versions
  vectorCUDA<float> dw; //For backprop
public:
  vector<int> indexLearnerIndices; //Variable to deliver indices in use for "double minibatch gradient descent"
  int nFeaturesIn;
  int nFeaturesOut;
  float dropout;
  IndexLearnerLayer(int nFeaturesIn, int nFeaturesOut) :
    nFeaturesIn(nFeaturesIn), nFeaturesOut(nFeaturesOut) {
    //float scale=pow(6.0f/(nFeaturesIn+nFeaturesOut),0.5f);
    W.resize (nFeaturesIn*nFeaturesOut); W.setZero();//Uniform(-scale,scale);
    MW.resize (nFeaturesIn*nFeaturesOut); MW.setZero();
    cout << "Index Learner " << nFeaturesIn << "-> " << nFeaturesOut << endl;
  }
  void preprocess
  (SpatiallySparseBatchInterface &input,
   SpatiallySparseBatchInterface &output) {
    assert(input.nFeatures==nFeaturesIn);
    assert(input.type==TRAINBATCH);
    output.type=input.type;
    output.batchSize=input.batchSize;
    output.nFeatures=nFeaturesOut;
    output.spatialSize=input.spatialSize;
    output.nSpatialSites=input.nSpatialSites;
    output.grids=input.grids;
    output.backgroundNullVectorNumbers=input.backgroundNullVectorNumbers;
    output.backpropErrors=true;
  }
  void forwards
  (SpatiallySparseBatchInterface &input,
   SpatiallySparseBatchInterface &output) {
    output.featuresPresent.hVector()=indexLearnerIndices;
    output.features.resize(output.nSpatialSites*output.featuresPresent.size());
    output.dfeatures.resize(output.nSpatialSites*output.featuresPresent.size());
    w.resize(input.featuresPresent.size()*output.featuresPresent.size());
    dShrinkMatrixForDropout<<<input.featuresPresent.size(),KERNELBLOCKSIZE>>>(W.dPtr(), w.dPtr(),
                                                                              input.featuresPresent.dPtr(),
                                                                              output.featuresPresent.dPtr(),
                                                                              output.nFeatures,
                                                                              output.featuresPresent.size());
    cudaCheckError();
    d_rowMajorSGEMM_alphaAB_betaC(cublasHandle,
                                  input.features.dPtr(), w.dPtr(), output.features.dPtr(),
                                  output.nSpatialSites, input.featuresPresent.size(), output.featuresPresent.size(),
                                  1.0f, 0.0f,__FILE__,__LINE__);
    applySigmoid(output, output, SOFTMAX);
    cudaCheckError();
  }
  void backwards(SpatiallySparseBatchInterface &input,
                 SpatiallySparseBatchInterface &output,
                 float learningRate=0.1) {
    applySigmoidBackProp(output, output, SOFTMAX);
    input.dfeatures.resize(input.nSpatialSites*input.featuresPresent.size());
    dw.resize(input.featuresPresent.size()*output.featuresPresent.size());
    d_rowMajorSGEMM_alphaAtB_betaC(cublasHandle,
                                   input.features.dPtr(), output.dfeatures.dPtr(), dw.dPtr(),
                                   input.featuresPresent.size(), output.nSpatialSites, output.featuresPresent.size(),
                                   1.0, 0.0);
    cudaCheckError();

    if (input.backpropErrors) {
      d_rowMajorSGEMM_alphaABt_betaC(cublasHandle,
                                     output.dfeatures.dPtr(), w.dPtr(), input.dfeatures.dPtr(),
                                     output.nSpatialSites,output.featuresPresent.size(),input.featuresPresent.size(),
                                     1.0, 0.0);
      cudaCheckError();
    }
    dGradientDescentShrunkMatrix<<<input.featuresPresent.size(),KERNELBLOCKSIZE>>>
      (dw.dPtr(), MW.dPtr(), W.dPtr(),
       output.nFeatures, output.featuresPresent.size(),
       input.featuresPresent.dPtr(), output.featuresPresent.dPtr(),
       learningRate);
    // output.features.resize(0);
    // output.dfeatures.resize(0);
    // cudaCheckError();
  }
  void loadWeightsFromStream(ifstream &f) {
    f.read((char*)&W.hVector()[0],sizeof(float)*W.size());
  };
  void putWeightsToStream(ofstream &f)  {
    f.write((char*)&W.hVector()[0],sizeof(float)*W.size());
  };
  int calculateInputSpatialSize(int outputSpatialSize) {
    return outputSpatialSize;
  }
};

void IndexLearner(SpatiallySparseBatchInterface& input, SpatiallySparseBatch& batch, int nTop) {
  assert(input.batchSize==input.nSpatialSites);
  assert(input.batchSize*input.batchSize==input.features.size());
  assert(input.type==TRAINBATCH);

  float* probs=&input.features.hVector()[0];
  for (int i=0;i<input.batchSize;++i)
    batch.probabilities.push_back(vector<float> (probs+i*input.batchSize,probs+(i+1)*input.batchSize));
  for (int i=0;i<input.batchSize;i++)
    batch.predictions.push_back(vectorTopIndices(batch.probabilities[i],nTop));

  batch.mistakes+=input.batchSize;
  for (int i=0;i<input.batchSize;i++) {
    batch.negativeLogLikelihood-=log(max(batch.probabilities[i][i],1.0e-15));
    for (int j=0;j<nTop;j++) {
      if (batch.predictions[i][j]==i) {
        batch.mistakes--;
      }
    }
  }
  //Begin backprop. Top layer: d Cost / d SoftmaxInput
  vectorCUDA<int> labels;
  labels.hVector()=range(input.batchSize);
  input.dfeatures.resize(input.nSpatialSites*input.featuresPresent.size());
  dDerivativeOfCostWRTpreSoftmaxTopLevelWeights<<<1,NTHREADS>>>
    (input.batchSize, input.dfeatures.dPtr(), input.features.dPtr(),
     labels.dPtr(), input.batchSize);
  cudaCheckError();
}
