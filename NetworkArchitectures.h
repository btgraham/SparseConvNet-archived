// Function:
// addLeNetLayerMP(int nFeatures,
//                 int filterSize, int filterStride,
//                 int poolSize, int poolStride,  //does nothing if poolSize==1
//                 ActivationFunction activationFn=RELU,
//                 float dropout=0.0f,
//                 bool maxPool=true) //average pooling otherwise


class FullyConnectedNet : public SpatiallySparseCNN {
public:
  FullyConnectedNet (int size, int n, ActivationFunction activationFn, int nInputFeatures, int nClasses, float p=0.0f, int cudaDevice=0, int nTop=1) : SpatiallySparseCNN(nInputFeatures, nClasses, cudaDevice, nTop) {
    addConvolutionalLayer(n,size,1,activationFn,p);
    addNetworkInNetworkLayer(n,activationFn,p);
    addNetworkInNetworkLayer(n,activationFn,p);
    addSoftmaxLayer();
  }
};

class LeNet5 : public SpatiallySparseCNN {
public:
  LeNet5 (int size=28,
          int nInputFeatures=1,
          ActivationFunction activationFn=RELU,
          int nClasses=10,
          int cudaDevice=0,
          int nTop=1) :
    SpatiallySparseCNN(nInputFeatures, nClasses, cudaDevice, nTop) { //size normally 28 or 32
    assert(size%4==0 and size>=16);
    addLeNetLayerMP(20,5,1,2,2,activationFn);
    addLeNetLayerMP(50,5,1,2,2,activationFn);
    addLeNetLayerMP(500,size/4-3,1,1,1,activationFn);
    addSoftmaxLayer();
  }
};

class DeepCNet : public SpatiallySparseCNN {
public:
  DeepCNet (int l, int k,
            ActivationFunction activationFn,
            int nInputFeatures,
            int nClasses,
            float p=0.0f,
            int cudaDevice=0,
            int nTop=1) :
    SpatiallySparseCNN(nInputFeatures, nClasses, cudaDevice, nTop) {
    for (int i=0;i<=l;i++)
      addLeNetLayerMP((i+1)*k,
                      (i==0)?3:2,
                      1,
                      (i<l)?2:1,
                      (i<l)?2:1,
                      activationFn,
                      p*i*1.0f/l);
    addSoftmaxLayer();
  }
};


class DeepCNiN : public SpatiallySparseCNN {
public:
  DeepCNiN(int l, int k,
           ActivationFunction activationFn,
           int nInputFeatures, int nClasses,
           float p=0.0f, int cudaDevice=0, int nTop=1) :
    SpatiallySparseCNN(nInputFeatures, nClasses, cudaDevice, nTop) {
    for (int i=0;i<=l;i++) {
      addLeNetLayerMP((i+1)*k,
                      (i==0)?3:2,
                      1,
                      (i<l)?2:1,
                      (i<l)?2:1,
                      activationFn, //!!!!!
                      p*i*1.0f/l);
      addNetworkInNetworkLayer((i+1)*k,activationFn,p*i*1.0f/l); //!!!
    }
    addSoftmaxLayer();
  }
};
class DeepC2C2 : public SpatiallySparseCNN {
public:
  DeepC2C2(int l, int k,
           ActivationFunction activationFn,
           int nInputFeatures, int nClasses,
           float p=0.0f, int cudaDevice=0, int nTop=1) :
    SpatiallySparseCNN(nInputFeatures, nClasses, cudaDevice, nTop) {
    for (int i=0;i<l;i++) {
      addLeNetLayerMP((i+1)*k,2,1,1,1,activationFn,p*i/(l+1));
      addLeNetLayerMP((i+1)*k,2,1,3,2,activationFn,p*i/(l+1));
    }
    addLeNetLayerMP((l+1)*k,2,1,1,1,activationFn,p*l/(l+1));
    addLeNetLayerMP((l+1)*k,2,1,1,1,activationFn,p*l/(l+1));
    addLeNetLayerMP((l+1)*k,1,1,1,1,activationFn,p);
    addSoftmaxLayer();
  }
};
class DeepC2C2C2 : public SpatiallySparseCNN {
public:
  DeepC2C2C2(int l, int k,
             ActivationFunction activationFn,
             int nInputFeatures, int nClasses,
             float p=0.0f, int cudaDevice=0, int nTop=1) :
    SpatiallySparseCNN(nInputFeatures, nClasses, cudaDevice, nTop) {
    for (int i=0;i<l;i++) {
      addLeNetLayerMP((i+1)*k,2,1,1,1,activationFn,p*i/(l+1));
      addLeNetLayerMP((i+1)*k,2,1,1,1,activationFn,p*i/(l+1));
      addLeNetLayerMP((i+1)*k,2,1,3,2,activationFn,p*i/(l+1));
    }
    addLeNetLayerMP((l+1)*k,2,1,1,1,activationFn,p*l/(l+1));
    addLeNetLayerMP((l+1)*k,2,1,1,1,activationFn,p*l/(l+1));
    addLeNetLayerMP((l+1)*k,2,1,1,1,activationFn,p*l/(l+1));
    addLeNetLayerMP((l+1)*k,1,1,1,1,activationFn,p);
    addSoftmaxLayer();
  }
};

class DeepC3C3C3 : public SpatiallySparseCNN {
public:
  DeepC3C3C3(int l, int k,
             ActivationFunction activationFn,
             int nInputFeatures, int nClasses, float p=0.0f, int cudaDevice=0, int nTop=1) :
    SpatiallySparseCNN(nInputFeatures, nClasses, cudaDevice, nTop) {
    for (int i=0;i<l;i++) {
      addLeNetLayerMP((i+1)*k,3,1,1,1,activationFn,p*i/(l+1));
      addLeNetLayerMP((i+1)*k,3,1,1,1,activationFn,p*i/(l+1));
      addLeNetLayerMP((i+1)*k,3,1,3,2,activationFn,p*i/(l+1));
    }
    addLeNetLayerMP((l+1)*k,3,1,1,1,activationFn,p*l/(l+1));
    addLeNetLayerMP((l+1)*k,3,1,1,1,activationFn,p*l/(l+1));
    addLeNetLayerMP((l+1)*k,3,1,1,1,activationFn,p*l/(l+1));
    addLeNetLayerMP((l+1)*k,1,1,1,1,activationFn,p);
    addSoftmaxLayer();
  }
};
