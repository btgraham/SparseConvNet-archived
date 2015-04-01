#define MAXPS2 1024

void selectGridPoolingRules
(vector<int> &g1, vector<int> &g2,
 int &bg1, int &bg2,
 int& nOutputSpatialSites,
 int s1, int s2, PoolingRegions& regions, int ps2 /*Squared region size */,
 vector<int>& rules) {
  assert(s2==1);
  g2.resize(1,-1);
  for (int ii=regions.i(0,0);ii<regions.I(0,0);ii++) {
    for (int jj=regions.j(0,0);jj<regions.J(0,0);jj++) {
      int n1=ii*s1+jj;
      if (g1[n1]!=bg1) {
        if(g2[0]==-1)
          g2[0]=nOutputSpatialSites++;
        rules.push_back(g1[n1]);
      }
    }
  }
  if (g2[0]==-1) {
    cout << "Empty select pooling\n";
    g2[0]=bg2=nOutputSpatialSites++;
    rules.push_back(bg1);
  }
  while (rules.size()%ps2!=0) rules.push_back(-1);
}

__global__ void dSelectPool(float* g1, float* g2, int* rules, int nOut, int ps2) {
  __shared__ int r[MAXPS2];  //Allocate at least size ps2 !!!!!!!!!!!
  int i=blockIdx.x*nOut;//for output g2
  for (int p=threadIdx.x;p<ps2;p+=KERNELBLOCKSIZE)
    r[p]=rules[blockIdx.x*ps2+p]*nOut;  //for input g1
  __syncthreads();
  for (int j=threadIdx.x;j<nOut;j+=KERNELBLOCKSIZE) { //nOut is a multiple of KERNELBLOCKSIZE!!!
    float t=0;
    int p=0;
    for (;p<ps2 and r[p]>=0;p++) {
      t+=g1[r[p]+j];
    }
    g2[i+j]=t/p;
  }
}

void selectPool(float* g1, float* g2, int* rules, int count, int ps2, int nOut) {
  int processed=0;
  assert(ps2<=MAXPS2);// if ps2>KERNELBLOCKSIZE, i.e. if poolSize>=5, allocate r more memory in dSelectPool and dSelectPoolBackProp
  while (processed<count) {
    int batch=min(32768,count-processed);
    dSelectPool<<<batch,KERNELBLOCKSIZE>>> (g1, g2+processed*nOut, rules+processed*ps2, nOut, ps2);
    processed+=batch;
  }
  cudaCheckError();
}


__global__ void dSelectPoolBackProp(float* d1, float* d2, int* rules, int nOut, int ps2) {
  __shared__ int r[MAXPS2];  //Allocate at least size ps2 !!!!!!!!!!!
  int i=blockIdx.x*nOut;//for input d2
  for (int p=threadIdx.x;p<ps2;p+=KERNELBLOCKSIZE) {
    r[p]=rules[blockIdx.x*ps2+p]*nOut;  //for output d1
  }
  __syncthreads();
  int maxP=0;
  while (maxP<ps2 and r[maxP]>=0)
    ++maxP;
  __syncthreads();
  for (int j=threadIdx.x;j<nOut;j+=KERNELBLOCKSIZE) {
    float t=d2[i+j]/maxP;
    for (int p=0;p<maxP;p++) {
      d1[r[p]+j]=t;
    }
  }
}

void selectPoolBackProp(float* d1, float* d2, int* rules, int count, int nOut, int ps2) {
  int processed=0;
  while (processed<count) {
    int batch=min(32768,count-processed);
    dSelectPoolBackProp<<<batch,KERNELBLOCKSIZE>>> (d1, d2+processed*nOut, rules+processed*ps2, nOut, ps2);
    processed+=batch;
  }
  cudaCheckError();
}

class SelectPoolingLayer : public SpatiallySparseLayer {
  int ps2;
public:
  int inSpatialSize;
  int outSpatialSize;
  int poolSize;
  int poolStride;
  SelectPoolingLayer(int poolSize) :
  poolSize(poolSize), poolStride(poolSize), ps2(poolSize*poolSize) {
  }
  void preprocess(SpatiallySparseBatchInterface &input,SpatiallySparseBatchInterface &output) {
    assert(input.spatialSize>=poolSize);
    assert((input.spatialSize-poolSize)%poolStride==0);
    output.type=input.type;
    output.batchSize=input.batchSize;
    output.nFeatures=input.nFeatures;
    output.featuresPresent.hVector()=input.featuresPresent.hVector();
    output.spatialSize=(input.spatialSize-poolSize)/poolStride+1;
    output.nSpatialSites=0;
    output.grids.resize(output.batchSize);
    output.backgroundNullVectorNumbers.resize(output.batchSize,-1);
    output.backpropErrors=input.backpropErrors;
    RegularPoolingRegions regions(inSpatialSize, outSpatialSize,poolSize, poolStride);
    for (int item=0;item<output.batchSize;item++)
      selectGridPoolingRules
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
  void forwards(SpatiallySparseBatchInterface &input,SpatiallySparseBatchInterface &output) {
    output.features.resize(output.nSpatialSites*output.featuresPresent.size());
    selectPool(input.features.dPtr(),output.features.dPtr(),output.rules.dPtr(),output.nSpatialSites,ps2,output.featuresPresent.size());
    cudaCheckError();
  }
  void backwards(SpatiallySparseBatchInterface &input,
                 SpatiallySparseBatchInterface &output,
                 float learningRate) {
    if (input.backpropErrors) {
      input.dfeatures.resize(input.nSpatialSites*input.featuresPresent.size());
      input.dfeatures.setZero();
      selectPoolBackProp(input.dfeatures.dPtr(), output.dfeatures.dPtr(), output.rules.dPtr(), output.nSpatialSites, output.featuresPresent.size(), ps2);
      // output.dfeatures.resize(0);
      // output.features.resize(0);
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
