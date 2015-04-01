int XConvType=1; //oscillate between 'x' and '+'

void XConvRules
(vector<int>& g0, vector<int>& g1,
 int& bg0, int& bg1,
 int& nOutputSpatialSites,
 int s0, int s1,
 int filterSize, //odd number
 int filterStride,
 vector<int>& rules,
 int f // 0/'x' or 1/'+'
 ) {
  bool needForNullVectorFoundYet=false;
  g1.resize(s1*s1,-1);
  for (int i=0;i<s1;i++) {
    for (int j=0;j<s1;j++) {
      int hotSites=0;
      int n1=i*s1+j;
      for (int z=0;z<2*filterSize-1;z++) {
        int ii,jj;
        if(f==1) {
          ii=(filterSize/2)+((z<filterSize)?(z-(filterSize/2)):0);
          jj=((filterSize/2)+((z<filterSize)?0:(z+1)))%filterSize;
        } else {
          ii=((filterSize/2)+z+1)%filterSize;
          jj=((filterSize/2)+((z<filterSize)?(z+1):(2*filterSize-z-1)))%filterSize;
        }
        int n0=(i*filterStride+ii)*s0+(j*filterStride+jj);
        if (g0[n0]!=bg0)
          hotSites++;
      }
      if (hotSites>filterSize) {
        g1[n1]=nOutputSpatialSites++;
        for (int z=0;z<2*filterSize-1;z++) {
          int ii,jj;
          if(f==1) {
            ii=(filterSize/2)+((z<filterSize)?(z-(filterSize/2)):0);
            jj=((filterSize/2)+((z<filterSize)?0:(z+1)))%filterSize;
          } else {
            ii=((filterSize/2)+z+1)%filterSize;
            jj=((filterSize/2)+((z<filterSize)?(z+1):(2*filterSize-z-1)))%filterSize;
          }
          int n0=(i*filterStride+ii)*s0+(j*filterStride+jj);
          rules.push_back(g0[n0]);
        }
      } else if (!needForNullVectorFoundYet) {
        g1[n1]=bg1=nOutputSpatialSites++;
        needForNullVectorFoundYet=true;
        for (int i=0; i<2*filterSize-1; i++)
          rules.push_back(bg0);
      } else {
        g1[n1]=bg1;
      }
    }
  }
  // int ctr=0;
  // for (int i=0;i<s1;i++)
  //   for (int j=0;j<s1;j++)
  //     if (g1[i*s1+j]!=bg1)
  //       ctr++;
  // cout <<s1 << " " <<flush;
  // if (s1==11) cout <<powf(ctr,0.5)<< " " << flush;
}

class XConvLayer : public SpatiallySparseLayer {
private:
  int fs;
public:
  int filterSize;
  int filterStride;
  int nFeaturesIn;
  int nFeaturesOut;
  int f;
  XConvLayer(int filterSize,
             int filterStride,
             int nFeaturesIn) :
    filterSize(filterSize),
    filterStride(filterStride),
    nFeaturesIn(nFeaturesIn) {
    f=XConvType=1-XConvType; //oscillate between 'x' and '+'
    fs=2*filterSize-1;
    nFeaturesOut=fs*nFeaturesIn;
    cout << "XConvolution (2*"
         << filterSize <<"-1)x" << nFeaturesIn
         << "->" << nFeaturesOut;
    if (filterStride>1)
      cout << " stride " << filterStride;
    cout <<endl;
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
      XConvRules(input.grids[item],
                 output.grids[item],
                 input.backgroundNullVectorNumbers[item],
                 output.backgroundNullVectorNumbers[item],
                 output.nSpatialSites,
                 input.spatialSize,
                 output.spatialSize,
                 filterSize, filterStride,
                 output.rules.hVector(),
                 f);
    }
    output.featuresPresent.copyToCPU();
    output.featuresPresent.resize(input.featuresPresent.size()*fs);
    convolutionFeaturesPresent(input.featuresPresent.hVector(), output.featuresPresent.hVector(), input.nFeatures, input.featuresPresent.size(), fs);
  }
  void forwards
  (SpatiallySparseBatchInterface &input,
   SpatiallySparseBatchInterface &output) {
    output.features.resize(output.nSpatialSites*output.featuresPresent.size());
    propForwardToMatrixMultiply(input.features.dPtr(),
                                output.features.dPtr(),
                                output.rules.dPtr(),
                                output.nSpatialSites*fs,
                                input.featuresPresent.size());
    cudaCheckError();
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
                                     output.nSpatialSites*fs,
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
