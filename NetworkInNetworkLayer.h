__global__ void dShrinkMatrixForDropout
(float* m, float* md,
 int* inFeaturesPresent, int* outFeaturesPresent,
 int nOut, int nOutDropout) {
  int i=blockIdx.x*nOutDropout;
  int ii=inFeaturesPresent[blockIdx.x]*nOut;
  for(int j=threadIdx.x; j<nOutDropout; j+=KERNELBLOCKSIZE) {
    int jj=outFeaturesPresent[j];
    md[i+j]=m[ii+jj];
  }
}

__global__ void dShrinkVectorForDropout(float* m, float* md, int* outFeaturesPresent, int nOut, int nOutDropout) {
  for(int i=threadIdx.x; i<nOutDropout; i+=NTHREADS) {
    md[i]=m[outFeaturesPresent[i]];
  }
}


#ifndef NAG_MU
#define NAG_MU 0.99
#endif

__global__ void dGradientDescent
(float* d_delta, float* d_momentum, float* d_weights, int nOut, float learningRate) {
  int i=blockIdx.x*nOut;
  for(int j=i+threadIdx.x; j<i+nOut; j+=KERNELBLOCKSIZE) {
    d_weights[j]-=d_momentum[j]*NAG_MU;
    d_momentum[j]=NAG_MU*d_momentum[j]-learningRate*(1-NAG_MU)*d_delta[j];
    d_weights[j]=d_weights[j]+d_momentum[j]*(1+NAG_MU);
  }
}

__global__ void dGradientDescentShrunkMatrix
(float* d_delta, float* d_momentum, float* d_weights,
 int nOut, int nOutDropout,
 int* inFeaturesPresent, int* outFeaturesPresent,
 float learningRate) {
  int i=blockIdx.x*nOutDropout;
  int ii=inFeaturesPresent[blockIdx.x]*nOut;
  for(int j=threadIdx.x; j<nOutDropout; j+=KERNELBLOCKSIZE) {
    int jj=outFeaturesPresent[j];
    //NAG light
    d_weights[ii+jj]-=d_momentum[ii+jj]*NAG_MU;
    d_momentum[ii+jj]=NAG_MU*d_momentum[ii+jj]-learningRate*(1-NAG_MU)*d_delta[i+j];
    d_weights[ii+jj]=d_weights[ii+jj]+d_momentum[ii+jj]*(1+NAG_MU);
  }
}

__global__ void dGradientDescentShrunkVector
(float* d_delta, float* d_momentum, float* d_weights,
 int nOut, int nOutDropout,
 int* outFeaturesPresent,
 float learningRate) {
  for(int i=threadIdx.x; i<nOutDropout; i+=NTHREADS) {
    int ii=outFeaturesPresent[i];
    //NAG light
    d_weights[ii]-=d_momentum[ii]*NAG_MU;
    d_momentum[ii]=NAG_MU*d_momentum[ii]-learningRate*(1-NAG_MU)*d_delta[i];
    d_weights[ii]=d_weights[ii]+d_momentum[ii]*(1+NAG_MU);
  }
}


__global__ void dColumnSum
(float* matrix, float* target, int nRows, int nColumns) {
  int i=blockIdx.x*KERNELBLOCKSIZE+threadIdx.x;
  float t=0;
  for (int j=blockIdx.y;j<nRows;j+=KERNELBLOCKSIZE)
    t+=matrix[j*nColumns+i];
  atomicAdd(&target[i],t);
}
void columnSum(float* matrix, float* target, int nRows, int nColumns) {
  if (nColumns/KERNELBLOCKSIZE>0)
    dColumnSum<<<dim3(nColumns/KERNELBLOCKSIZE,KERNELBLOCKSIZE),KERNELBLOCKSIZE>>>(matrix, target, nRows, nColumns);
  if (nColumns%KERNELBLOCKSIZE>0) {
    int o=nColumns/KERNELBLOCKSIZE*KERNELBLOCKSIZE;
    dColumnSum<<<dim3(1,KERNELBLOCKSIZE),nColumns-o>>>(matrix+o, target+o, nRows, nColumns);
  }
  cudaCheckError();
}

__global__ void dReplicateArray
(float* src, float* dst, int nColumns) {
  int i=blockIdx.x*nColumns;
  for (int j=threadIdx.x;j<nColumns;j+=KERNELBLOCKSIZE)
    dst[i+j]=src[j];
}
void replicateArray(float* src, float* dst, int nRows, int nColumns) {
  int processed=0;
  while (processed<nRows) {
    int batch=min(1024,nRows-processed); //////////////////////////////////////
    dReplicateArray<<<batch,KERNELBLOCKSIZE>>>
      (src, dst+processed*nColumns, nColumns);
    processed+=batch;
  }
  cudaCheckError();
}

class NetworkInNetworkLayer : public SpatiallySparseLayer {
  RNG rng;
public:
  vectorCUDA<float> W; //Weights
  vectorCUDA<float> MW; //momentum
  vectorCUDA<float> w; //shrunk versions
  vectorCUDA<float> dw; //For backprop
#if BIAS == 1
  vectorCUDA<float> B; //Weights
  vectorCUDA<float> MB; //momentum
  vectorCUDA<float> b; //shrunk versions
  vectorCUDA<float> db; //For backprop
#endif
  ActivationFunction fn;
  int nFeaturesIn;
  int nFeaturesOut;
  float dropout;
  NetworkInNetworkLayer(int nFeaturesIn, int nFeaturesOut,
                        float dropout=0,ActivationFunction fn=NOSIGMOID,
                        float alpha=1//used to determine intialization weights only
                        ) :
    nFeaturesIn(nFeaturesIn), nFeaturesOut(nFeaturesOut),
    dropout(dropout), fn(fn) {
    float scale=pow(6.0f/(nFeaturesIn+nFeaturesOut*alpha),0.5f);
    W.resize (nFeaturesIn*nFeaturesOut); W.setUniform(-scale,scale);
    MW.resize (nFeaturesIn*nFeaturesOut); MW.setZero();
#if BIAS == 1
    B.resize (nFeaturesOut); B.setZero();
    MB.resize (nFeaturesOut); MB.setZero();
#endif
    cout << "Learn " << nFeaturesIn << "->" << nFeaturesOut << " dropout=" << dropout << " " << sigmoidNames[fn] <<endl;
  }
  void preprocess
  (SpatiallySparseBatchInterface &input,
   SpatiallySparseBatchInterface &output) {
    output.type=input.type;
    output.batchSize=input.batchSize;
    output.nFeatures=nFeaturesOut;
    assert(input.nFeatures==nFeaturesIn);
    output.spatialSize=input.spatialSize;
    output.nSpatialSites=input.nSpatialSites;
    output.grids=input.grids;
    output.backgroundNullVectorNumbers=input.backgroundNullVectorNumbers;
    int o=nFeaturesOut*(input.type==TRAINBATCH?(1.0f-dropout):1.0f);
    output.featuresPresent.hVector()=rng.NchooseM(nFeaturesOut,o);
    output.backpropErrors=true;
  }
  void forwards
  (SpatiallySparseBatchInterface &input,
   SpatiallySparseBatchInterface &output) {
    output.features.resize(output.nSpatialSites*output.featuresPresent.size());
    if (input.type==TRAINBATCH and
        nFeaturesIn+nFeaturesOut>input.featuresPresent.size()+output.featuresPresent.size()) {
      w.resize(input.featuresPresent.size()*output.featuresPresent.size());
      dShrinkMatrixForDropout<<<input.featuresPresent.size(),KERNELBLOCKSIZE>>>(W.dPtr(), w.dPtr(),
                                                                                input.featuresPresent.dPtr(),
                                                                                output.featuresPresent.dPtr(),
                                                                                output.nFeatures,
                                                                                output.featuresPresent.size());
      cudaCheckError();


#if BIAS == 1
      b.resize(output.featuresPresent.size());
      dShrinkVectorForDropout<<<1,NTHREADS>>>(B.dPtr(), b.dPtr(),
                                              output.featuresPresent.dPtr(),
                                              output.nFeatures,
                                              output.featuresPresent.size());
      cudaCheckError();
      replicateArray(b.dPtr(), output.features.dPtr(), output.nSpatialSites, output.featuresPresent.size());
      cudaCheckError();
      d_rowMajorSGEMM_alphaAB_betaC(cublasHandle,
                                    input.features.dPtr(), w.dPtr(), output.features.dPtr(),
                                    output.nSpatialSites, input.featuresPresent.size(), output.featuresPresent.size(),
                                    1.0f, 1.0f,__FILE__,__LINE__);

#else
      d_rowMajorSGEMM_alphaAB_betaC(cublasHandle,
                                    input.features.dPtr(), w.dPtr(), output.features.dPtr(),
                                    output.nSpatialSites, input.featuresPresent.size(), output.featuresPresent.size(),
                                    1.0f, 0.0f,__FILE__,__LINE__);
#endif
      cudaCheckError();

    } else {
#if BIAS == 1
      replicateArray(B.dPtr(), output.features.dPtr(), output.nSpatialSites, output.featuresPresent.size());
      d_rowMajorSGEMM_alphaAB_betaC(cublasHandle,
                                    input.features.dPtr(), W.dPtr(), output.features.dPtr(),
                                    output.nSpatialSites, input.nFeatures, output.nFeatures,
                                    1.0f-dropout, 1.0f-dropout,__FILE__,__LINE__);
#else
      d_rowMajorSGEMM_alphaAB_betaC(cublasHandle,
                                    input.features.dPtr(), W.dPtr(), output.features.dPtr(),
                                    output.nSpatialSites, input.nFeatures, output.nFeatures,
                                    1.0f-dropout, 0.0f,__FILE__,__LINE__);
#endif
      cudaCheckError();
    }
    applySigmoid(output, output, fn);
    cudaCheckError();
  }
  void scaleWeights(SpatiallySparseBatchInterface &input,
                    SpatiallySparseBatchInterface &output) {
    assert(input.features.size()>0);
    assert(output.features.size()>0); //call after forwards(...)
    float s=output.features.meanAbs(); cout << "out scale " << s << endl;
    W.multiplicativeRescale(powf(s,-0.1));
    B.multiplicativeRescale(powf(s,-0.1));
    MW.multiplicativeRescale(powf(s,-0.1));
    MB.multiplicativeRescale(powf(s,-0.1));
  }
  void backwards(SpatiallySparseBatchInterface &input,
                 SpatiallySparseBatchInterface &output,
                 float learningRate=0.1) {
    applySigmoidBackProp(output, output, fn);

    dw.resize(input.featuresPresent.size()*output.featuresPresent.size());
#if BIAS == 1
    db.resize(output.featuresPresent.size());
#endif

    d_rowMajorSGEMM_alphaAtB_betaC(cublasHandle,
                                   input.features.dPtr(), output.dfeatures.dPtr(), dw.dPtr(),
                                   input.featuresPresent.size(), output.nSpatialSites, output.featuresPresent.size(),
                                   1.0, 0.0);
    cudaCheckError();
#if BIAS == 1
    db.setZero();
    columnSum(output.dfeatures.dPtr(), db.dPtr(), output.nSpatialSites, output.featuresPresent.size());
#endif

    if (nFeaturesIn+nFeaturesOut>input.featuresPresent.size()+output.featuresPresent.size()) {
      if (input.backpropErrors) {
        input.dfeatures.resize(input.nSpatialSites*input.featuresPresent.size());
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

#if BIAS == 1
      dGradientDescentShrunkVector<<<1,NTHREADS>>>
        (db.dPtr(), MB.dPtr(), B.dPtr(),
         output.nFeatures, output.featuresPresent.size(),
         output.featuresPresent.dPtr(),
         learningRate);
#endif
    } else {
      if (input.backpropErrors) {
        input.dfeatures.resize(input.nSpatialSites*input.featuresPresent.size());
        d_rowMajorSGEMM_alphaABt_betaC(cublasHandle,
                                       output.dfeatures.dPtr(), W.dPtr(), input.dfeatures.dPtr(),
                                       output.nSpatialSites,nFeaturesOut,nFeaturesIn,
                                       1.0, 0.0);
        cudaCheckError();
      }
      dGradientDescent<<<nFeaturesIn,KERNELBLOCKSIZE>>>
        (dw.dPtr(), MW.dPtr(), W.dPtr(),  nFeaturesOut, learningRate);
#if BIAS == 1
      dGradientDescent<<<1,KERNELBLOCKSIZE>>>
        (db.dPtr(), MB.dPtr(), B.dPtr(), nFeaturesOut, learningRate);
#endif
    }
    // output.features.resize(0);
    // output.dfeatures.resize(0);
    // cudaCheckError();
  }
  void loadWeightsFromStream(ifstream &f) {
    f.read((char*)&W.hVector()[0],sizeof(float)*W.size());
#if BIAS == 1
    f.read((char*)&B.hVector()[0],sizeof(float)*B.size());
#endif
  };
  void putWeightsToStream(ofstream &f)  {
    f.write((char*)&W.hVector()[0],sizeof(float)*W.size());
#if BIAS == 1
    f.write((char*)&B.hVector()[0],sizeof(float)*B.size());
#endif
  };
  int calculateInputSpatialSize(int outputSpatialSize) {
    return outputSpatialSize;
  }
};
