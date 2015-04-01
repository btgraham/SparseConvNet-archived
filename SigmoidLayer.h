enum ActivationFunction      {NOSIGMOID, RELU,   VLEAKYRELU,   LEAKYRELU, TANH,  SOFTMAX};
const char *sigmoidNames[] ={ ""       , "ReLU", "VeryLeakyReLU", "LeakyReLU", "tanh", "Softmax Classification"};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void dSigmoidReLu
(float* a, float* b, int nOut) {
  int i=blockIdx.x*nOut;
  for (int j=i+threadIdx.x; j<i+nOut; j+=KERNELBLOCKSIZE) {
    b[j]=(a[j]>0)?a[j]:0;
  }
}
void sigmoidReLU(float* a, float* b, int count, int nOut) {
  int processed=0;
  while (processed<count) {
    int batch=min(32768/4,count-processed);
    dSigmoidReLu<<<batch,KERNELBLOCKSIZE>>>
      (a+processed*nOut, b+processed*nOut, nOut);
    processed+=batch;
  }
  cudaCheckError();
}

__global__ void dSigmoidBackpropReLu
(float* a, float* b, float* da, float* db, int nOut) {
  int i=blockIdx.x*nOut;
  for (int j=i+threadIdx.x; j<i+nOut; j+=KERNELBLOCKSIZE) {
    da[j]=(a[j]>0)?db[j]:0;
  }
}
void sigmoidBackpropReLU(float* a, float* b, float* da, float* db, int count, int nOut) {
  int processed=0;
  while (processed<count) {
    int batch=min(32768/4,count-processed);
    dSigmoidBackpropReLu<<<batch,KERNELBLOCKSIZE>>>
      (a+processed*nOut, b+processed*nOut, da+processed*nOut, db+processed*nOut, nOut);
    processed+=batch;
  }
  cudaCheckError();
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void dSigmoidTanh
(float* a, float* b, int nOut) {
  int i=blockIdx.x*nOut;
  for (int j=i+threadIdx.x; j<i+nOut; j+=KERNELBLOCKSIZE) {
    b[j]=tanhf(a[j]);
  }
}
void sigmoidTanh(float* a, float* b, int count, int nOut) {
  int processed=0;
  while (processed<count) {
    int batch=min(32768/4,count-processed);
    dSigmoidTanh<<<batch,KERNELBLOCKSIZE>>>
      (a+processed*nOut, b+processed*nOut, nOut);
    processed+=batch;
  }
  cudaCheckError();
}

__global__ void dSigmoidBackpropTanh
(float* a, float* b, float* da, float* db, int nOut) {
  int i=blockIdx.x*nOut;
  for (int j=i+threadIdx.x; j<i+nOut; j+=KERNELBLOCKSIZE) {
    da[j]=db[j]*(1+b[j])*(1-b[j]);
  }
}
void sigmoidBackpropTanh(float* a, float* b, float* da, float* db, int count, int nOut) {
  int processed=0;
  while (processed<count) {
    int batch=min(32768/4,count-processed);
    dSigmoidBackpropTanh<<<batch,KERNELBLOCKSIZE>>>
      (a+processed*nOut, b+processed*nOut, da+processed*nOut, db+processed*nOut, nOut);
    processed+=batch;
  }
  cudaCheckError();
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void dSigmoidLeakyReLu
(float* a, float* b, int nOut, float alpha) {
  int i=blockIdx.x*nOut;
  for (int j=i+threadIdx.x; j<i+nOut; j+=KERNELBLOCKSIZE) {
    b[j]=(a[j]>0)?a[j]:(a[j]*alpha);
  }
}
void sigmoidLeakyReLU(float* a, float* b, int count, int nOut, float alpha=0.01) {
  int processed=0;
  while (processed<count) {
    int batch=min(32768,count-processed);
    dSigmoidLeakyReLu<<<batch,KERNELBLOCKSIZE>>>
      (a+processed*nOut, b+processed*nOut, nOut,alpha);
    processed+=batch;
  }
  cudaCheckError();
}

__global__ void dSigmoidBackpropLeakyReLu
(float* a, float* b, float* da, float* db, int nOut, float alpha) {
  int i=blockIdx.x*nOut;
  for (int j=i+threadIdx.x; j<i+nOut; j+=KERNELBLOCKSIZE) {
    da[j]=(a[j]>0)?db[j]:(db[j]*alpha);
    __syncthreads();
  }
}
void sigmoidBackpropLeakyReLU(float* a, float* b, float* da, float* db, int count, int nOut, float alpha=0.01) {
  int processed=0;
  while (processed<count) {
    int batch=min(32768,count-processed);
    dSigmoidBackpropLeakyReLu<<<batch,KERNELBLOCKSIZE>>>
      (a+processed*nOut, b+processed*nOut, da+processed*nOut, db+processed*nOut, nOut,alpha);
    processed+=batch;
  }
  cudaCheckError();
}




//SOFTMAX only occurs at the top layer;
//derivative contained in calculation of initial d_delta.
__global__ void dSigmoidSoftmax(float* a, float* b, int count, int nOut) {
  for(int i=threadIdx.x; i<count; i+=NTHREADS) {
    float acc=0.0f;
    float mx=-10000.0f;
    for (int k=0;k<nOut;k++)
      if (a[i*nOut+k]>mx) mx=a[i*nOut+k];
    for (int k=0;k<nOut;k++) {
      b[i*nOut+k]=expf((a[i*nOut+k]-mx)); //Subtract row max value for numerical stability.
      acc+=b[i*nOut+k];}
    for (int k=0;k<nOut;k++) {
      b[i*nOut+k]/=acc;
    }
  }
}
__global__ void dSigmoidBackpropSoftmax(float* a, float* b, float* da, float* db, int count, int nOut) {
  for(int i=0; i<count; i++) {
    for (int k=threadIdx.x; k<nOut; k+=NTHREADS) {
      da[i*nOut+k]=db[i*nOut+k];
    }
  }
}
void applySigmoid(SpatiallySparseBatchInterface& input, SpatiallySparseBatchInterface& output, ActivationFunction fn) {
  //  cout << fn << endl;
  switch(fn) {
  case TANH:
    sigmoidTanh
      (input.features.dPtr(),
       output.features.dPtr(),
       output.nSpatialSites,
       output.featuresPresent.size());
    break;
  case RELU:
    sigmoidReLU
      (input.features.dPtr(),
       output.features.dPtr(),
       output.nSpatialSites,
       output.featuresPresent.size());
    break;
  case LEAKYRELU:
    sigmoidLeakyReLU
      (input.features.dPtr(),
       output.features.dPtr(),
       output.nSpatialSites,
       output.featuresPresent.size(),
       0.01);
    break;
  case VLEAKYRELU:
    sigmoidLeakyReLU
      (input.features.dPtr(),
       output.features.dPtr(),
       output.nSpatialSites,
       output.featuresPresent.size(),
       0.333);
    break;
  case SOFTMAX:
    dSigmoidSoftmax      <<<1,NTHREADS>>> (input.features.dPtr(),output.features.dPtr(),output.nSpatialSites,output.featuresPresent.size());
    break;
  case NOSIGMOID:
    break;
  }
}

void applySigmoidBackProp(SpatiallySparseBatchInterface& input, SpatiallySparseBatchInterface& output, ActivationFunction fn) {
  switch(fn) {
  case TANH:
    sigmoidBackpropTanh
      (input.features.dPtr(),output.features.dPtr(),
       input.dfeatures.dPtr(),
       output.dfeatures.dPtr(),
       output.nSpatialSites,
       output.featuresPresent.size());
    break;
  case RELU:
    sigmoidBackpropReLU
      (input.features.dPtr(),output.features.dPtr(),
       input.dfeatures.dPtr(),
       output.dfeatures.dPtr(),
       output.nSpatialSites,
       output.featuresPresent.size());
    break;
  case LEAKYRELU:
    sigmoidBackpropLeakyReLU
      (input.features.dPtr(),
       output.features.dPtr(),
       input.dfeatures.dPtr(),
       output.dfeatures.dPtr(),
       output.nSpatialSites,
       output.featuresPresent.size(),
       0.01);
    break;
  case VLEAKYRELU:
    sigmoidBackpropLeakyReLU
      (input.features.dPtr(),
       output.features.dPtr(),
       input.dfeatures.dPtr(),
       output.dfeatures.dPtr(),
       output.nSpatialSites,
       output.featuresPresent.size(),
       0.333);
    break;
  case SOFTMAX:
    dSigmoidBackpropSoftmax  <<<1,NTHREADS>>>
      (input.features.dPtr(),output.features.dPtr(), input.dfeatures.dPtr(),output.dfeatures.dPtr(), output.nSpatialSites, output.featuresPresent.size());   break;
  case NOSIGMOID:
    break;
  }
}

class SigmoidLayer : public SpatiallySparseLayer {
public:
  ActivationFunction fn;
  SigmoidLayer(ActivationFunction fn) : fn(fn) {
    cout << sigmoidNames[fn]<<endl;
  };
  void preprocess
  (SpatiallySparseBatchInterface &input,
   SpatiallySparseBatchInterface &output) {
    output.type=input.type;
    output.batchSize=input.batchSize;
    output.nFeatures=input.nFeatures;
    output.featuresPresent.hVector()=input.featuresPresent.hVector();
    output.spatialSize=input.spatialSize;
    output.nSpatialSites=input.nSpatialSites;
    output.grids=input.grids;
    output.backgroundNullVectorNumbers=input.backgroundNullVectorNumbers;
    output.backpropErrors=input.backpropErrors;
  }
  void forwards(SpatiallySparseBatchInterface &input,SpatiallySparseBatchInterface &output) {
    output.features.resize(output.nSpatialSites*output.featuresPresent.size());
    applySigmoid(input, output, fn);
  }
  void backwards(SpatiallySparseBatchInterface &input,
                 SpatiallySparseBatchInterface &output,
                 float learningRate) {
    if (input.backpropErrors) {
      input.dfeatures.resize(input.nSpatialSites*input.featuresPresent.size());
      applySigmoidBackProp(input, output, fn);
      // output.features.resize(0);
      // output.dfeatures.resize(0);
      // cudaCheckError();
    }
  }
  int calculateInputSpatialSize(int outputSpatialSize) {
    return outputSpatialSize;
  }
};
