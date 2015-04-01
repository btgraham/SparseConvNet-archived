// x4-convolution. Alternate between
//         1...1                                ..1..
//         .2.2.                                ..2..
//         ..4..          and                   12421
//         .2.2.                                ..2..
//         1...1                                ..1..

// layer numbered 0,1,...,n-1
// reinterpret as 0,1,...,4(n-1)   (need multiple of 128=4*KERNELBLOCKSIZE units)

// 0,0; 4,0; 0,4; 4,4 ::: 0
// 1,1; 1,3; 3,1; 3,3 ::: 0,1
// 2,2                ::: 0,1,2,3

// 2,0; 0,2; 2,4; 4,2 ::: 0
// 2,1; 1,2; 3,2; 2,3 ::: 0,1
// 2,2                ::: 0,1,2,3

int X4I[]={-2,+2,-2,+2,-1,-1,-1,-1,+1,+1,+1,+1, 0, 0, 0, 0,  +2,-2, 0, 0,+1,+1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0}; //2x16 rules per new site
int X4J[]={-2,-2,+2,+2,-1,-1,+1,+1,-1,-1,+1,+1, 0, 0, 0, 0,   0, 0,+2,-2, 0, 0, 0, 0,+1,+1,-1,-1, 0, 0, 0, 0};
int X4K[]={ 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3,   0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3};


void X4ConvRules
(vector<int>& g0, vector<int>& g1,
 int& bg0, int& bg1,
 int& nOutputSpatialSites,
 int s,
 vector<int>& rules,
 int f // 0=='x' or 1=='+'
 ) {
  bool needForNullVectorFoundYet=false;
  g1.resize(s*s,-1);
  for (int i1=0;i1<s;i1++) {
    for (int j1=0;j1<s;j1++) {
      int n1=i1*s+j1;
      if (g0[n1]!=bg0) {
        g1[n1]=nOutputSpatialSites++;
        for (int z=16*f;z<16*(f+1);z++) {
          int i0=i1+X4I[z];
          int j0=j1+X4J[z];
          if (0<= i0 and i0<s and 0<=j0 and j0<s)
            rules.push_back(4*g0[i0*s+j0]+X4K[z]);
          else
            rules.push_back(4*bg0+X4K[z]);
        }
      } else if (!needForNullVectorFoundYet) {
        g1[n1]=bg1=nOutputSpatialSites++;
        needForNullVectorFoundYet=true;
        for (int z=0; z<16; z++)
          rules.push_back(4*bg0+X4K[z]);
      } else {
        g1[n1]=bg1;
      }
    }
  }
}

class X4ConvLayer : public SpatiallySparseLayer {
public:
  int nFeaturesIn;
  int nFeaturesOut;
  int f;
  X4ConvLayer(int nFeaturesIn) :
  nFeaturesIn(nFeaturesIn) {
    f=XConvType=1-XConvType; //oscillate between 'x' and '+'
    nFeaturesOut=4*nFeaturesIn;
    cout << "X4Convolution 16/4x"
         << nFeaturesIn
         << "->"
         << nFeaturesOut
         << endl;


  }
  void preprocess
  (SpatiallySparseBatchInterface &input,
   SpatiallySparseBatchInterface &output) {
    output.type=input.type;
    output.batchSize=input.batchSize;
    output.nFeatures=nFeaturesOut;
    assert(input.featuresPresent.size()==input.nFeatures); //assume no dropout yet
    assert(input.nFeatures==nFeaturesIn);
    output.spatialSize=input.spatialSize;
    output.nSpatialSites=0;
    output.grids.resize(output.batchSize);
    output.backgroundNullVectorNumbers.resize(output.batchSize,-1);
    output.backpropErrors=input.backpropErrors;
    for (int item=0;item<output.batchSize;item++) {
      X4ConvRules(input.grids[item],
                  output.grids[item],
                  input.backgroundNullVectorNumbers[item],
                  output.backgroundNullVectorNumbers[item],
                  output.nSpatialSites,
                  input.spatialSize,
                  output.rules.hVector(),
                  f);
    }
    output.featuresPresent.copyToCPU();
    output.featuresPresent.hVector()=range(nFeaturesOut);
  }
  void forwards
  (SpatiallySparseBatchInterface &input,
   SpatiallySparseBatchInterface &output) {
    output.features.resize(output.nSpatialSites*output.featuresPresent.size());
    propForwardToMatrixMultiply(input.features.dPtr(),
                                output.features.dPtr(),
                                output.rules.dPtr(),
                                output.nSpatialSites*16,
                                input.nFeatures/4);
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
                                     output.nSpatialSites*16,
                                     input.nFeatures/4);
      // output.features.resize(0);
      // output.dfeatures.resize(0);
      // cudaCheckError();
    }
  }
  int calculateInputSpatialSize(int outputSpatialSize) {
    return outputSpatialSize;
  }
};
