#define AVERAGE_POOLING_MAX_PS2 64

__global__ void dAveragePool(float* g1, float* g2, int* rules, int nOut, int ps2) {
  __shared__ int r[AVERAGE_POOLING_MAX_PS2];  //Allocate at least size ps2
  int i=blockIdx.x*nOut;//for output g2
  for (int p=threadIdx.x;p<ps2;p+=KERNELBLOCKSIZE)
    r[p]=rules[blockIdx.x*ps2+p]*nOut;  //for input g1
  __syncthreads();
  for (int j=threadIdx.x;j<nOut;j+=KERNELBLOCKSIZE) {
    float acc=0;
    for (int p=0;p<ps2;p++) {
      acc+=g1[r[p]+j];
    }
    g2[i+j]=acc/ps2;
  }
}

void averagePool(float* g1, float* g2, int* rules, int count, int ps2, int nOut) {
  int processed=0;
  assert(ps2<=AVERAGE_POOLING_MAX_PS2);
  while (processed<count) {
    int batch=min(32768,count-processed);
    dAveragePool<<<batch,KERNELBLOCKSIZE>>> (g1, g2+processed*nOut, rules+processed*ps2, nOut, ps2);
    processed+=batch;
  }
  cudaCheckError();
}


__global__ void dAveragePoolBackProp(float* d1, float* d2, int* rules, int nOut, int ps2) {
  __shared__ int r[AVERAGE_POOLING_MAX_PS2];
  int i=blockIdx.x*nOut;//for input d2
  for (int p=threadIdx.x;p<ps2;p+=KERNELBLOCKSIZE)
    r[p]=rules[blockIdx.x*ps2+p]*nOut;  //for output d1
  __syncthreads();
  for (int j=threadIdx.x;j<nOut;j+=KERNELBLOCKSIZE) {
    float deriv=d2[i+j]/ps2;
    for (int p=0;p<ps2;p++) {
      atomicAdd(&d1[r[p]+j],deriv);//d1[r[p]+j]=deriv;
    }
  }
}

void averagePoolBackProp(float* d1, float* d2, int* rules, int count, int nOut, int ps2) {
  int processed=0;
  while (processed<count) {
    int batch=min(32768,count-processed);
    dAveragePoolBackProp<<<batch,KERNELBLOCKSIZE>>> (d1, d2+processed*nOut, rules+processed*ps2, nOut, ps2);
    processed+=batch;
  }
  cudaCheckError();
}

class AveragePoolingLayer : public SpatiallySparseLayer {
  int ps2;
public:
  int inSpatialSize;
  int outSpatialSize;
  int poolSize;
  int poolStride;
  AveragePoolingLayer(int poolSize, int poolStride) :
    poolSize(poolSize), poolStride(poolStride), ps2(poolSize*poolSize) {
    cout << "AveragePooling " << poolSize << " " << poolStride << endl;
  }
  void preprocess
  (SpatiallySparseBatchInterface &input,
   SpatiallySparseBatchInterface &output) {
    output.type=input.type;
    output.batchSize=input.batchSize;
    output.nFeatures=input.nFeatures;
    output.featuresPresent.hVector()=input.featuresPresent.hVector();
    assert(input.spatialSize>=poolSize);
    assert((input.spatialSize-poolSize)%poolStride==0);
    output.spatialSize=(input.spatialSize-poolSize)/poolStride+1;
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
  }
  void forwards(SpatiallySparseBatchInterface &input,SpatiallySparseBatchInterface &output) {
    output.features.resize(output.nSpatialSites*output.featuresPresent.size());
    averagePool(input.features.dPtr(),output.features.dPtr(),output.rules.dPtr(),output.nSpatialSites,ps2,output.featuresPresent.size());
    cudaCheckError();
  }
  void backwards(SpatiallySparseBatchInterface &input,
                 SpatiallySparseBatchInterface &output,
                 float learningRate) {
    if (input.backpropErrors) {
      input.dfeatures.resize(input.nSpatialSites*input.featuresPresent.size());
      input.dfeatures.setZero();
      averagePoolBackProp(input.dfeatures.dPtr(), output.dfeatures.dPtr(), output.rules.dPtr(), output.nSpatialSites, output.featuresPresent.size(), ps2);
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
