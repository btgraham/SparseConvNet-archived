int YConvType=0; //oscillate between 'Y' and '⅄'
//
// .01234567   .01234567
// 0x.......   0x......x                            x           x         x
// 1........   1..x...x.                                          x     x
// 2.x......   2....xx..                            x               x x
// 3........   3.....x..          ->                                 x
// 4..x.....   4........                            x      and
// 5..xx....   5......x.                           x x               x
// 6.x...x..   6........                         x     x
// 7x......x   7.......x                       x         x           x
//
int YConvII[]={0,1,3,3,4,1,5,0,7,   0,7,2,6,4,5,5,6,7};
int YConvJJ[]={0,2,4,5,5,6,6,7,7,   0,0,1,1,2,2,3,5,7};


void YConvRules
(vector<int>& g0, vector<int>& g1,
 int& bg0, int& bg1,
 int& nOutputSpatialSites,
 int s0, int s1,
 int filterSize,
 int filterStride,
 vector<int>& rules,
 int f //// 0 or 1; alternate between 'Y' and '⅄'
 ) {
  assert(filterSize==8);
  bool needForNullVectorFoundYet=false;
  g1.resize(s1*s1,-1);
  for (int i=0;i<s1;i++) {
    for (int j=0;j<s1;j++) {
      int hotSites=0;
      int n1=i*s1+j;
      for (int z=9*f;z<9*(f+1);z++) {
        int n0=(i*filterStride+YConvII[z])*s0+(j*filterStride+YConvJJ[z]);
        if (g0[n0]!=bg0)
          hotSites++;
      }
      if (hotSites>=4) {
        g1[n1]=nOutputSpatialSites++;
        for (int z=9*f;z<9*(f+1);z++) {
          int n0=(i*filterStride+YConvII[z])*s0+(j*filterStride+YConvJJ[z]);
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
}

class YConvLayer : public SpatiallySparseLayer {
private:
  int fs;
public:
  int filterSize;
  int filterStride;
  int nFeaturesIn;
  int nFeaturesOut;
  int f;
  int activityThreshold;
  YConvLayer(int filterStride,
             int nFeaturesIn) :
    filterSize(8),
    filterStride(filterStride),
    nFeaturesIn(nFeaturesIn),
    activityThreshold(activityThreshold) {
    f=YConvType=1-YConvType; //oscillate between Y and upside-down-Y
    fs=9;
    nFeaturesOut=fs*nFeaturesIn;
    cout << "YConvolution 9x"
         << nFeaturesIn
         << "->"
         << nFeaturesOut;
    if (filterStride>1)
      cout << " stride " << filterStride;
    cout <<endl;

  }
  void preprocess
  (SpatiallySparseBatchInterface &input,
   SpatiallySparseBatchInterface &output) {
    assert(input.nFeatures==nFeaturesIn);
    assert(input.spatialSize>=filterSize);
    assert((input.spatialSize-filterSize)%filterStride==0);
    output.type=input.type;
    output.batchSize=input.batchSize;
    output.nFeatures=nFeaturesOut;
    output.spatialSize=(input.spatialSize-filterSize)/filterStride+1;
    output.nSpatialSites=0;
    output.grids.resize(output.batchSize);
    output.backgroundNullVectorNumbers.resize(output.batchSize,-1);
    output.backpropErrors=input.backpropErrors;
    for (int item=0;item<output.batchSize;item++) {
      YConvRules(input.grids[item],
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
