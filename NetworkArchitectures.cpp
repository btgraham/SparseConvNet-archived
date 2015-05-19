#include "NetworkArchitectures.h"

DeepCNet::DeepCNet (int dimension, int l, int k, ActivationFunction fn, int nInputFeatures, int nClasses, float p, int cudaDevice, int nTop) : SparseConvNet(dimension,nInputFeatures, nClasses, cudaDevice, nTop) {
  for (int i=0;i<=l;i++)
    addLeNetLayerMP((i+1)*k,
                    (i==0)?3:2,
                    1,
                    (i<l)?3:1,
                    (i<l)?2:1,
                    fn,
                    p*i*1.0f/l);
  addSoftmaxLayer();
}

DeepC2::DeepC2 (int dimension, int l, int k, ActivationFunction fn, int nInputFeatures, int nClasses, float p, int cudaDevice, int nTop) : SparseConvNet(dimension,nInputFeatures, nClasses, cudaDevice, nTop) {
  for (int i=0;i<=l;i++)
    addLeNetLayerMP((i+1)*k,
                    2,
                    1,
                    (i<l)?3:1,
                    (i<l)?2:1,
                    fn,
                    p*i*1.0f/l);
  addSoftmaxLayer();
}

DeepCNiN::DeepCNiN (int dimension, int l, int k, ActivationFunction fn, int nInputFeatures, int nClasses, float p, int cudaDevice, int nTop) : SparseConvNet(dimension,nInputFeatures, nClasses, cudaDevice, nTop) {
  for (int i=0;i<=l;i++) {
    addLeNetLayerMP((i+1)*k,
                    (i==0)?2:2,
                    1,
                    (i<l)?3:1,
                    (i<l)?2:1,
                    fn,
                    p*i*1.0f/l);
    addLeNetLayerMP(1,1,1,1,(i+1)*k,fn,p*i*1.0f/l);
  }
  addSoftmaxLayer();
}
DeepC2C2::DeepC2C2(int dimension, int l, int k, ActivationFunction fn, int nInputFeatures, int nClasses, float p, int cudaDevice, int nTop) : SparseConvNet(dimension,nInputFeatures, nClasses, cudaDevice, nTop) {
  for (int i=0;i<l;i++) {
    addLeNetLayerMP((i+1)*k,2,1,1,1,fn,p*i*1.0f/l);
    addLeNetLayerMP((i+1)*k,2,1,3,2,fn,p*i*1.0f/l);
  }
  addLeNetLayerMP((l+1)*k,2,1,1,1,fn,p);
  addLeNetLayerMP((l+1)*k,1,1,1,1,fn,p);
  addSoftmaxLayer();
}
