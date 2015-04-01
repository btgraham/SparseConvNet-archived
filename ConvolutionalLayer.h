__global__ void dPropForwardToMatrixMultiplyInput
(float* d_features, float* d_convolved, int* rules, int count, int nIn) {
  __shared__ int r[KERNELBLOCKSIZE];
  for (int c=blockIdx.x*KERNELBLOCKSIZE; c<count; c+=(1<<12)*KERNELBLOCKSIZE) {
    int n=min(KERNELBLOCKSIZE,count-c);
    r[threadIdx.x]=(threadIdx.x<n)?rules[c+threadIdx.x]*nIn:0;
    __syncthreads();
    for (int q=0;q<n;q++) {
      int rq=r[q];
      int i=(c+q)*nIn;
      for (int j=threadIdx.x;j<nIn;j+=KERNELBLOCKSIZE) {
        d_convolved[i+j]=d_features[rq+j];
      }
    }
    __syncthreads();
  }
}
void propForwardToMatrixMultiply(float* inFeatures, float* outFeatures, int* rules, int count, int nIn) {
  int batch=min(1<<12,(count+KERNELBLOCKSIZE-1)/KERNELBLOCKSIZE);
  dPropForwardToMatrixMultiplyInput<<<batch,KERNELBLOCKSIZE>>>
    (inFeatures,
     outFeatures,
     rules,
     count, nIn);
  cudaCheckError();
}
__global__ void dPropBackwardFromMatrixMultiplyOutput
(float* d_deltaGrid, float* d_deltaConvolved, int* rules, int count, int nIn) {
  __shared__ int r[KERNELBLOCKSIZE];
  for (int c=blockIdx.x*KERNELBLOCKSIZE; c<count; c+=(1<<12)*KERNELBLOCKSIZE) {
    int n=min(KERNELBLOCKSIZE,count-c);
    r[threadIdx.x]=(threadIdx.x<n)?rules[c+threadIdx.x]*nIn:0;
    __syncthreads();
    for (int q=0;q<n;q++) {
      int rq=r[q];
      int i=(c+q)*nIn;
      for (int j=threadIdx.x;j<nIn;j+=KERNELBLOCKSIZE) {
        if (d_deltaConvolved[i+j]!=0)
          atomicAdd(&d_deltaGrid[rq+j],d_deltaConvolved[i+j]);
      }
    }
    __syncthreads();
  }
}
void propBackwardFromMatrixMultiply(float* inDFeatures, float* outDFeatures, int* rules, int count, int nIn) {
  int batch=min(1<<12,(count+KERNELBLOCKSIZE-1)/KERNELBLOCKSIZE);
  dPropBackwardFromMatrixMultiplyOutput<<<batch,KERNELBLOCKSIZE>>>
    (inDFeatures,
     outDFeatures,
     rules,
     count, nIn);
  cudaCheckError();
}


void gridConvolutionRules
(vector<int>& g0, vector<int>& g1,
 int& bg0, int& bg1,
 int& nOutputSpatialSites,
 int s0, int s1,
 int filterSize, int filterStride,
 vector<int>& rules) {
  bool needForNullVectorFoundYet=false;
  g1.resize(s1*s1,-1);
  for (int i=0;i<s1;i++) {
    for (int j=0;j<s1;j++) {
      int n1=i*s1+j;
      for (int ii=0;ii<filterSize;ii++) {
        for (int jj=0;jj<filterSize;jj++) {
          int n0=(i*filterStride+ii)*s0+(j*filterStride+jj);
          if (g0[n0]!=bg0 && g1[n1]==-1)
            g1[n1]=nOutputSpatialSites++;
        }
      }
      if (g1[n1]==-1 and !needForNullVectorFoundYet) {
        bg1=nOutputSpatialSites++;
        needForNullVectorFoundYet=true;
        for (int i=0; i<filterSize*filterSize; i++)
          rules.push_back(bg0);
      }
      if (g1[n1]==-1)
        g1[n1]=bg1;
      else {
        for (int ii=0;ii<filterSize;ii++) {
          for (int jj=0;jj<filterSize;jj++) {
            int n0=(i*filterStride+ii)*s0+(j*filterStride+jj);
            rules.push_back(g0[n0]);
          }
        }
      }
    }
  }
}

template <typename t> void convolutionFeaturesPresent(vector<t>& d_src, vector<t>& d_dest, int nf, int nfp, int nCopies) {
  for (int i=0;i<nfp*nCopies;++i) {
    d_dest[i]=d_src[i%nfp]+nf*(i/nfp);
  }
}

class ConvolutionalLayer : public SpatiallySparseLayer {
private:
  int fs2;
public:
  int filterSize;
  int filterStride;  //Normally equals 1 if filterSize is small. Could be taken to be 3 or 4 if filterSize is larger.
  int nFeaturesIn;
  int nFeaturesOut;
  ConvolutionalLayer(int filterSize,
                     int filterStride,
                     int nFeaturesIn) :
    filterSize(filterSize),
    filterStride(filterStride),
    nFeaturesIn(nFeaturesIn) {
    fs2=filterSize*filterSize;
    nFeaturesOut=fs2*nFeaturesIn;
    cout << "Convolution "
         << filterSize <<"x" <<filterSize<< "x"<< nFeaturesIn
         << "->" << nFeaturesOut;
    if (filterStride>1)
      cout << " stride " << filterStride;
    cout << endl;

  }
  void preprocess
  (SpatiallySparseBatchInterface &input,
   SpatiallySparseBatchInterface &output) {
    output.type=input.type;
    output.batchSize=input.batchSize;
    output.nFeatures=nFeaturesOut;
    assert(input.nFeatures==nFeaturesIn);
    assert(input.spatialSize>=filterSize);
    assert((input.spatialSize-filterSize)%filterStride==0);
    output.spatialSize=(input.spatialSize-filterSize)/filterStride+1;
    output.nSpatialSites=0;
    output.grids.resize(output.batchSize);
    output.backgroundNullVectorNumbers.resize(output.batchSize,-1);
    output.backpropErrors=input.backpropErrors;
    for (int item=0;item<output.batchSize;item++) {
      gridConvolutionRules(input.grids[item],
                           output.grids[item],
                           input.backgroundNullVectorNumbers[item],
                           output.backgroundNullVectorNumbers[item],
                           output.nSpatialSites,
                           input.spatialSize,
                           output.spatialSize,
                           filterSize, filterStride,
                           output.rules.hVector());
    }
    output.featuresPresent.copyToCPU();
    output.featuresPresent.resize(input.featuresPresent.size()*fs2);
    convolutionFeaturesPresent(input.featuresPresent.hVector(), output.featuresPresent.hVector(), input.nFeatures, input.featuresPresent.size(), fs2);
  }
  void forwards
  (SpatiallySparseBatchInterface &input,
   SpatiallySparseBatchInterface &output) {
    output.features.resize(output.nSpatialSites*output.featuresPresent.size());
    propForwardToMatrixMultiply(input.features.dPtr(),
                                output.features.dPtr(),
                                output.rules.dPtr(),
                                output.nSpatialSites*fs2,
                                input.featuresPresent.size());
  }
  void backwards(SpatiallySparseBatchInterface &input,
                 SpatiallySparseBatchInterface &output,
                 float learningRate=0.1) {
    if (input.backpropErrors) {
      input.dfeatures.resize(input.nSpatialSites*input.featuresPresent.size());
      input.dfeatures.setZero();
      propBackwardFromMatrixMultiply(input.dfeatures.dPtr(),
                                     output.dfeatures.dPtr(),
                                     output.rules.dPtr(),
                                     output.nSpatialSites*fs2,
                                     input.featuresPresent.size());
      // output.features.resize(0);
      // output.dfeatures.resize(0);
      // cudaCheckError();
    }
  }
  int calculateInputSpatialSize(int outputSpatialSize) {
    return filterSize+(outputSpatialSize-1)*filterStride;
  }
};
