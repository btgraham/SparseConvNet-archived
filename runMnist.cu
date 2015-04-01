int epoch=0;       // 0 to start, positive to restart
int cudaDevice=-1; // GPU pciBusID, -1 for default
int batchSize=100;

#include "SparseConvNet.h"
#include "OpenCVPicture.h"
#include "readMNIST.h"

Picture* OpenCVPicture::distort(RNG& rng, batchType type) {
  OpenCVPicture* pic=new OpenCVPicture(*this);
  pic->loadData();
  if (type==TRAINBATCH) {
    float
      c00=1, c01=0,  //2x2 identity matrix---starting point for calculating affine distortion matrix
      c10=0, c11=1;
    c00*=1+rng.uniform(-0.2,0.2); // x stretch
    c11*=1+rng.uniform(-0.2,0.2); // y stretch
    int r=rng.randint(3);
    float alpha=rng.uniform(-0.2,0.2);
    if (r==0) matrixMul2x2inPlace(c00,c01,c10,c11,1,0,alpha,1); //Slant
    if (r==1) matrixMul2x2inPlace(c00,c01,c10,c11,1,alpha,0,1); //Slant other way
    if (r==2) matrixMul2x2inPlace(c00,c01,c10,c11,cos(alpha),-sin(alpha),sin(alpha),cos(alpha)); //Rotate
    transformImage(pic->mat, backgroundColor, c00, c01, c10, c11);
    pic->jiggle(rng,4);
  }
  return pic;
}

class LeNet : public SpatiallySparseCNN { //A version of Yann LeCun's LeNet - see http://yann.lecun.com/exdb/lenet/
public:
  LeNet (int nInputFeatures=1,
         int nClasses=10,
         int cudaDevice=0,
         int nTop=1) :
    SpatiallySparseCNN(nInputFeatures, nClasses, cudaDevice, nTop) {
    addLeNetLayerMP(16,5,1,3,2,RELU);
    addLeNetLayerMP(32,5,1,3,2,RELU);
    addLeNetLayerMP(256,5,1,1,1,RELU);
    addSoftmaxLayer();
  }
};

int main() {
  string baseName="weights/mnist_";

  SpatialDataset trainSet=MnistTrainSet();
  SpatialDataset testSet=MnistTestSet();
  trainSet.summary();
  testSet.summary();

  LeNet cnn(trainSet.nFeatures,trainSet.nClasses,cudaDevice);
  if (epoch++>0)
    cnn.loadWeights(baseName,epoch-1);
  for (;epoch<=100;epoch++) {
    cout <<"epoch: " << epoch << flush;
    cnn.processDataset(trainSet, batchSize,0.003*exp(-epoch*0.05));
    if (epoch%10==0) {
      cnn.processDataset(testSet,  batchSize);
      cnn.saveWeights(baseName,epoch);
    }
  }
}
