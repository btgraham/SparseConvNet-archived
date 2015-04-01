class PoolingRegions { //Location of the (i,j)-th pooling region
public:
  virtual int i(int i, int j)=0; //x-axis lower bound
  virtual int I(int i, int j)=0; //       upper
  virtual int j(int i, int j)=0; //y-axis lower bound
  virtual int J(int i, int j)=0; //       upper
};

class RegularPoolingRegions : public PoolingRegions {
  int nIn;
  int nOut;
  int poolSize;
  int poolStride;
public:
  RegularPoolingRegions(int nIn, int nOut, int poolSize, int poolStride) : nIn(nIn), nOut(nOut), poolSize(poolSize), poolStride(poolStride) {
    assert(nIn==poolSize+(nOut-1)*poolStride);
  }
  int i(int i, int j) {
    return i*poolStride;
  }
  int I(int i, int j) {
    return i*poolStride+poolSize;
  }
  int j(int i, int j) {
    return j*poolStride;
  }
  int J(int i, int j) {
    return j*poolStride+poolSize;
  }
};

class PseudorandomOverlappingFractionalMaxPoolingBlocks {
public:
  vector<int> b0;
  vector<int> b1;
  vector<int> s;
  PseudorandomOverlappingFractionalMaxPoolingBlocks(int nIn, int nOut, RNG& rng) {
    assert(nOut<nIn);
    b0.resize(nOut);
    b1.resize(nOut);
    float alpha=(nIn-2)*1.0/(nOut-1);
    float u=rng.uniform(0,10000);
    for (int i=1;i<nOut;++i)
      s.push_back( (int)((i+1+u)*alpha) - (int)((i+u)*alpha) );
    for (int i=0;i<nOut-1;i++) {
      b0[i+1]=b0[i]+s[i];
    }
    b0[nOut-1]=nIn-2;
    for (int i=0;i<nOut;i++) {
      b1[i]=b0[i]+2;
    }
  }
};

class PseudorandomOverlappingFractionalMaxPoolingRegions : public PoolingRegions {
  PseudorandomOverlappingFractionalMaxPoolingBlocks x;
  PseudorandomOverlappingFractionalMaxPoolingBlocks y;
public:
  PseudorandomOverlappingFractionalMaxPoolingRegions(int nIn, int nOut, RNG& rng) : x(nIn,nOut,rng), y(nIn, nOut,rng) {}
  int i(int i, int j) {
    return x.b0[i];
  }
  int I(int i, int j) {
    return x.b1[i];
  }
  int j(int i, int j) {
    return y.b0[j];
  }
  int J(int i, int j) {
    return y.b1[j];
  }
};

class RandomOverlappingFractionalMaxPoolingBlocks {
public:
  vector<int> b0;
  vector<int> b1;
  vector<int> s;
  RandomOverlappingFractionalMaxPoolingBlocks(int nIn, int nOut,RNG& rng) {
    b0.resize(nOut);
    b1.resize(nOut);
    for (int i=0;i<2*nOut-nIn;++i)
      s.push_back(1);
    while (s.size()<nOut-1)
      s.push_back(2);
    rng.vectorShuffle(s);
    s.push_back(2);
    for (int i=0;i<nOut-1;i++)
      b0[i+1]=b0[i]+s[i];
    for (int i=0;i<nOut;i++)
      b1[i]=b0[i]+2;
  }
};
class RandomOverlappingFractionalMaxPoolingRegions : public PoolingRegions {
  RandomOverlappingFractionalMaxPoolingBlocks x;
  RandomOverlappingFractionalMaxPoolingBlocks y;
public:
  RandomOverlappingFractionalMaxPoolingRegions(int nIn, int nOut, RNG& rng) : x(nIn,nOut,rng), y(nIn, nOut,rng) {}
  int i(int i, int j) {
    return x.b0[i];
  }
  int I(int i, int j) {
    return x.b1[i];
  }
  int j(int i, int j) {
    return y.b0[j];
  }
  int J(int i, int j) {
    return y.b1[j];
  }
};

class JitteryOverlappingFractionalMaxPoolingRegions : public PoolingRegions {
  RNG& rng;
  int nIn, nOut,poolSize;
  vector<int> ii, jj;
  int bound(float k) {
    return max(0,min(nIn-poolSize,(int)round(k)));
  }
public:
  JitteryOverlappingFractionalMaxPoolingRegions(int nIn, int nOut, int poolSize, RNG& rng) : nIn(nIn),nOut(nOut), poolSize(poolSize), rng(rng) {
    float alpha=(nIn-poolSize)/(nOut-1.0);
    ii.resize(nOut*nOut);
    jj.resize(nOut*nOut);
    for (int i=0;i<nOut;++i) {
      for (int j=0;j<nOut;++j) {
        ii[i*nOut+j]=bound(i*alpha+rng.stdNormal());
        jj[i*nOut+j]=bound(j*alpha+rng.stdNormal());
      }
    }
  }
  int i(int i, int j) {
    return ii[i*nOut+j];
  }
  int I(int i, int j) {
    return ii[i*nOut+j]+2;
  }
  int j(int i, int j) {
    return jj[i*nOut+j];
  }
  int J(int i, int j) {
    return jj[i*nOut+j]+2;
  }
};

void gridPoolingRules
(vector<int> &g1, vector<int> &g2,
 int &bg1, int &bg2,
 int& nOutputSpatialSites,
 int s1, int s2, PoolingRegions& regions, int ps2 /*Maximum (squared) region size */,
 vector<int>& rules) {
  bool needForNullVectorFoundYet=false;
  g2.resize(s2*s2,-1);
  for (int i=0;i<s2;i++) {
    for (int j=0;j<s2;j++) {
      int n2=i*s2+j;
      for (int ii=regions.i(i,j);ii<regions.I(i,j);ii++) {
        for (int jj=regions.j(i,j);jj<regions.J(i,j);jj++) {
          int n1=ii*s1+jj;
          if (g1[n1]!=bg1 && g2[n2]==-1)
            g2[n2]=nOutputSpatialSites++;
        }
      }
      if (g2[n2]==-1 and !needForNullVectorFoundYet) {
        bg2=nOutputSpatialSites++;
        needForNullVectorFoundYet=true;
        for (int i=0; i<ps2; i++)
          rules.push_back(bg1);
      }
      if (g2[n2]==-1)
        g2[n2]=bg2;
      else {
        for (int ii=regions.i(i,j);ii<regions.I(i,j);ii++) {
          for (int jj=regions.j(i,j);jj<regions.J(i,j);jj++) {
            int n1=ii*s1+jj;
            rules.push_back(g1[n1]);
          }
        }
        while (rules.size()%ps2!=0) rules.push_back(-1); //To allow for pooling regions smaller than ps2
      }
    }
  }
}
#define MAX_PS2 25
__global__ void dMaxPool(float* g1, float* g2, int* rules, int nOut, int* d_choice, int ps2) {
  __shared__ int r[MAX_PS2]; //Cache rules
  int i=blockIdx.x*nOut;//for output
  for (int p=threadIdx.x;p<ps2;p+=KERNELBLOCKSIZE)
    r[p]=rules[blockIdx.x*ps2+p]*nOut;  //for input
  __syncthreads();
  for (int j=threadIdx.x;j<nOut;j+=KERNELBLOCKSIZE) {
    float s,t;
    int c;
    for (int p=0;p<ps2;p++) {
      s=(r[p]>=0)?g1[r[p]+j]:-10000000;
      if (p==0 or t<s) {
        c=r[p]+j;
        t=s;
      }
    }
    g2[i+j]=t;
    d_choice[i+j]=c;
  }
}
void maxPool(float* g1, float* g2, int* rules, int count, int ps2, int nOut, int* d_choice) {
  assert(ps2<MAX_PS2);
  int processed=0;
  while (processed<count) {
    int batch=min(32768,count-processed);
    dMaxPool<<<batch,KERNELBLOCKSIZE>>> (g1, g2+processed*nOut, rules+processed*ps2, nOut, d_choice+processed*nOut,ps2);
    processed+=batch;
  }
  cudaCheckError();
}

__global__ void dMaxPoolBackProp
(float* d1, float* d2, int nOut,int* d_choice) {
  //initialize d1 to zero first!!!
  int i=blockIdx.x*nOut;
  for (int j=i+threadIdx.x;j<i+nOut;j+=KERNELBLOCKSIZE) {
    atomicAdd(&d1[d_choice[j]],d2[j]);    //d1[d_choice[j]]=d2[j];
  }
}

void maxPoolBackProp(float* d1, float* d2, int count, int nOut, int* d_choice) {
  int processed=0;
  while (processed<count) {
    int batch=min(32768,count-processed);
    dMaxPoolBackProp<<<batch,KERNELBLOCKSIZE>>> (d1, d2+processed*nOut, nOut, d_choice+processed*nOut);
    processed+=batch;
  }
}


//Refactor


class MaxPoolingLayer : public SpatiallySparseLayer {
  int ps2;
public:
  int inSpatialSize;
  int outSpatialSize;
  int poolSize;
  int poolStride;
  MaxPoolingLayer(int poolSize, int poolStride) : poolSize(poolSize), poolStride(poolStride), ps2(poolSize*poolSize) {
    cout << "MaxPooling " << poolSize << " " << poolStride << endl;
  }
  void preprocess
  (SpatiallySparseBatchInterface &input,
   SpatiallySparseBatchInterface &output) {
    assert(input.spatialSize==inSpatialSize);
    output.type=input.type;
    output.batchSize=input.batchSize;
    output.nFeatures=input.nFeatures;
    output.spatialSize=outSpatialSize;
    output.nSpatialSites=0;
    output.grids.resize(output.batchSize);
    output.backgroundNullVectorNumbers.resize(output.batchSize,-1);
    output.backpropErrors=input.backpropErrors;
    RegularPoolingRegions regions(inSpatialSize, outSpatialSize,poolSize, poolStride);
    for (int item=0;item<output.batchSize;item++)
      gridPoolingRules
        (input.grids[item],
         output.grids[item],
         input.backgroundNullVectorNumbers[item],
         output.backgroundNullVectorNumbers[item],
         output.nSpatialSites,
         input.spatialSize,
         output.spatialSize,
         regions,
         ps2,
         output.rules.hVector());
    output.featuresPresent.hVector()=input.featuresPresent.hVector();
  }
  void forwards(SpatiallySparseBatchInterface &input,SpatiallySparseBatchInterface &output) {
    output.poolingChoices.resize(output.nSpatialSites*output.featuresPresent.size());
    output.features.resize(output.nSpatialSites*output.featuresPresent.size());
    cudaCheckError();
    maxPool(input.features.dPtr(),output.features.dPtr(),output.rules.dPtr(),output.nSpatialSites,ps2,output.featuresPresent.size(),output.poolingChoices.dPtr());
    cudaCheckError();
  }
  void backwards(SpatiallySparseBatchInterface &input,
                 SpatiallySparseBatchInterface &output,
                 float learningRate) {
    if (input.backpropErrors) {
      input.dfeatures.resize(input.nSpatialSites*input.featuresPresent.size());
      input.dfeatures.setZero();
      maxPoolBackProp
        (input.dfeatures.dPtr(), output.dfeatures.dPtr(), output.nSpatialSites, output.featuresPresent.size(), output.poolingChoices.dPtr());
      // output.features.resize(0);
      // output.dfeatures.resize(0);
      // cudaCheckError();
    }
  }
  int calculateInputSpatialSize(int outputSpatialSize) {
    outSpatialSize=outputSpatialSize;
    inSpatialSize=poolSize+(outputSpatialSize-1)*poolStride;
    cout << "(" << outSpatialSize <<"," <<inSpatialSize << ") ";
    return inSpatialSize;
  }
};



class RandomOverlappingFractionalMaxPoolingLayer : public SpatiallySparseLayer {
  int ps2;
public:
  int inSpatialSize;
  int outSpatialSize;
  float fmpShrink;
  RNG rng;
  RandomOverlappingFractionalMaxPoolingLayer(float fmpShrink) : fmpShrink(fmpShrink), ps2(4) {
    cout << "RandomOverlappingFractionalMaxPooling " << fmpShrink << endl;
  }
  void preprocess
  (SpatiallySparseBatchInterface &input,
   SpatiallySparseBatchInterface &output) {
    assert(input.spatialSize==inSpatialSize);
    output.type=input.type;
    output.batchSize=input.batchSize;
    output.nFeatures=input.nFeatures;
    output.featuresPresent.hVector()=input.featuresPresent.hVector();
    output.spatialSize=outSpatialSize;
    output.nSpatialSites=0;
    output.grids.resize(output.batchSize);
    output.backgroundNullVectorNumbers.resize(output.batchSize,-1);
    output.backpropErrors=input.backpropErrors;
    for (int item=0;item<output.batchSize;item++) {
      RandomOverlappingFractionalMaxPoolingRegions regions(inSpatialSize, outSpatialSize,rng);
      gridPoolingRules
        (input.grids[item],
         output.grids[item],
         input.backgroundNullVectorNumbers[item],
         output.backgroundNullVectorNumbers[item],
         output.nSpatialSites,
         input.spatialSize,
         output.spatialSize,
         regions,
         ps2,
         output.rules.hVector());
    }
  }
  void forwards(SpatiallySparseBatchInterface &input,SpatiallySparseBatchInterface &output) {
    output.poolingChoices.resize(output.nSpatialSites*output.featuresPresent.size());
    output.features.resize(output.nSpatialSites*output.featuresPresent.size());
    // if (input.type==TRAINBATCH)
    //   output.dfeatures.resize(output.nSpatialSites*output.featuresPresent.size());
    cudaCheckError();
    maxPool(input.features.dPtr(),output.features.dPtr(),output.rules.dPtr(),output.nSpatialSites,ps2,output.featuresPresent.size(),output.poolingChoices.dPtr());
    cudaCheckError();
  }
  void backwards(SpatiallySparseBatchInterface &input,
                 SpatiallySparseBatchInterface &output,
                 float learningRate) {
    if (input.backpropErrors) {
      input.dfeatures.resize(input.nSpatialSites*input.featuresPresent.size());
      input.dfeatures.setZero();
      maxPoolBackProp
        (input.dfeatures.dPtr(), output.dfeatures.dPtr(), output.nSpatialSites, output.featuresPresent.size(), output.poolingChoices.dPtr());
      // output.features.resize(0);
      // output.dfeatures.resize(0);
      // cudaCheckError();
    }
  }
  int calculateInputSpatialSize(int outputSpatialSize) {
    outSpatialSize=outputSpatialSize;
    inSpatialSize=outputSpatialSize*fmpShrink+0.5;
    if (inSpatialSize==outputSpatialSize)
      inSpatialSize++;
    cout << "(" << outSpatialSize <<"," <<inSpatialSize << ") ";
    return inSpatialSize;
  }
};


class PseudorandomOverlappingFractionalMaxPoolingLayer : public SpatiallySparseLayer {
  int ps2;
public:
  int inSpatialSize;
  int outSpatialSize;
  float fmpShrink;
  RNG rng;
  PseudorandomOverlappingFractionalMaxPoolingLayer(float fmpShrink) : fmpShrink(fmpShrink), ps2(4) {
    cout << "PseudorandomOverlappingFractionalMaxPooling " << fmpShrink << endl;
  }
  void preprocess
  (SpatiallySparseBatchInterface &input,
   SpatiallySparseBatchInterface &output) {
    output.type=input.type;
    output.batchSize=input.batchSize;
    output.nFeatures=input.nFeatures;
    output.featuresPresent.hVector()=input.featuresPresent.hVector();
    assert(input.spatialSize==inSpatialSize);
    output.spatialSize=outSpatialSize;
    output.nSpatialSites=0;
    output.grids.resize(output.batchSize);
    output.backgroundNullVectorNumbers.resize(output.batchSize,-1);
    output.backpropErrors=input.backpropErrors;
    for (int item=0;item<output.batchSize;item++) {
      PseudorandomOverlappingFractionalMaxPoolingRegions regions(inSpatialSize, outSpatialSize,rng);
      gridPoolingRules
        (input.grids[item],
         output.grids[item],
         input.backgroundNullVectorNumbers[item],
         output.backgroundNullVectorNumbers[item],
         output.nSpatialSites,
         input.spatialSize,
         output.spatialSize,
         regions,
         ps2,
         output.rules.hVector());
    }
  }
  void forwards(SpatiallySparseBatchInterface &input,SpatiallySparseBatchInterface &output) {
    output.poolingChoices.resize(output.nSpatialSites*output.featuresPresent.size());
    output.features.resize(output.nSpatialSites*output.featuresPresent.size());
    // if (input.type==TRAINBATCH)
    //   output.dfeatures.resize(output.nSpatialSites*output.featuresPresent.size());
    cudaCheckError();
    maxPool(input.features.dPtr(),output.features.dPtr(),output.rules.dPtr(),output.nSpatialSites,ps2,output.featuresPresent.size(),output.poolingChoices.dPtr());
    cudaCheckError();
  }
  void backwards(SpatiallySparseBatchInterface &input,
                 SpatiallySparseBatchInterface &output,
                 float learningRate) {
    if (input.backpropErrors) {
      input.dfeatures.resize(input.nSpatialSites*input.featuresPresent.size());
      input.dfeatures.setZero();
      maxPoolBackProp
        (input.dfeatures.dPtr(), output.dfeatures.dPtr(), output.nSpatialSites, output.featuresPresent.size(), output.poolingChoices.dPtr());
      // output.features.resize(0);
      // output.dfeatures.resize(0);
      // cudaCheckError();
    }
  }
  int calculateInputSpatialSize(int outputSpatialSize) {
    outSpatialSize=outputSpatialSize;
    inSpatialSize=outputSpatialSize*fmpShrink+0.5;
    if (inSpatialSize==outputSpatialSize)
      inSpatialSize++;
    cout << "(" << outSpatialSize <<"," <<inSpatialSize << ") ";
    return inSpatialSize;
  }
};


class JitteryOverlappingFractionalMaxPoolingLayer : public SpatiallySparseLayer {
  int ps2;
public:
  int inSpatialSize;
  int outSpatialSize;
  int poolSize;
  float fmpShrink;
  RNG rng;
  JitteryOverlappingFractionalMaxPoolingLayer(float fmpShrink, int poolSize=2) : fmpShrink(fmpShrink), poolSize(poolSize), ps2(poolSize*poolSize) {
    cout << "JitteryOverlappingFractionalMaxPooling " << fmpShrink << endl;
  }
  void preprocess
  (SpatiallySparseBatchInterface &input,
   SpatiallySparseBatchInterface &output) {
    output.type=input.type;
    output.batchSize=input.batchSize;
    output.nFeatures=input.nFeatures;
    output.featuresPresent.hVector()=input.featuresPresent.hVector();
    assert(input.spatialSize==inSpatialSize);
    output.spatialSize=outSpatialSize;
    output.nSpatialSites=0;
    output.grids.resize(output.batchSize);
    output.backgroundNullVectorNumbers.resize(output.batchSize,-1);
    output.backpropErrors=input.backpropErrors;
    for (int item=0;item<output.batchSize;item++) {
      JitteryOverlappingFractionalMaxPoolingRegions regions(inSpatialSize, outSpatialSize,poolSize,rng);
      gridPoolingRules
        (input.grids[item],
         output.grids[item],
         input.backgroundNullVectorNumbers[item],
         output.backgroundNullVectorNumbers[item],
         output.nSpatialSites,
         input.spatialSize,
         output.spatialSize,
         regions,
         ps2,
         output.rules.hVector());
    }
  }
  void forwards(SpatiallySparseBatchInterface &input,SpatiallySparseBatchInterface &output) {
    output.poolingChoices.resize(output.nSpatialSites*output.featuresPresent.size());
    output.features.resize(output.nSpatialSites*output.featuresPresent.size());
    // if (input.type==TRAINBATCH)
    //   output.dfeatures.resize(output.nSpatialSites*output.featuresPresent.size());
    cudaCheckError();
    maxPool(input.features.dPtr(),output.features.dPtr(),output.rules.dPtr(),output.nSpatialSites,ps2,output.featuresPresent.size(),output.poolingChoices.dPtr());
    cudaCheckError();
  }
  void backwards(SpatiallySparseBatchInterface &input,
                 SpatiallySparseBatchInterface &output,
                 float learningRate) {
    if (input.backpropErrors) {
      input.dfeatures.resize(input.nSpatialSites*input.featuresPresent.size());
      input.dfeatures.setZero();
      maxPoolBackProp
        (input.dfeatures.dPtr(), output.dfeatures.dPtr(), output.nSpatialSites, output.featuresPresent.size(), output.poolingChoices.dPtr());
      // output.features.resize(0);
      // output.dfeatures.resize(0);
      // cudaCheckError();
    }
  }
  int calculateInputSpatialSize(int outputSpatialSize) {
    outSpatialSize=outputSpatialSize;
    inSpatialSize=poolSize+(outputSpatialSize-1)*fmpShrink+0.5;
    if (inSpatialSize==outputSpatialSize)
      inSpatialSize++;
    cout << "(" << outSpatialSize <<"," <<inSpatialSize << ") ";
    return inSpatialSize;
  }
};
