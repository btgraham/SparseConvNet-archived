#include "SparseConvNet.h"
#include "NetworkArchitectures.h"
#include "SpatiallySparseDatasetCIFAR10.h"

int epoch=0;
int cudaDevice=1; //PCI bus ID, -1 for default GPU
int batchSize=50;

Picture* OpenCVPicture::distort(RNG& rng, batchType type) {
  OpenCVPicture* pic=new OpenCVPicture(*this);
  pic->loadData();
  if (type==TRAINBATCH) {
    float
      c00=1, c01=0,  //2x2 identity matrix---starting point for calculating affine distortion matrix
      c10=0, c11=1;
    c00*=1+rng.uniform(-0.2,0.2); // x stretch
    c11*=1+rng.uniform(-0.2,0.2); // y stretch
    if (rng.randint(2)==0) c00*=-1; //Horizontal flip
    int r=rng.randint(3);
    float alpha=rng.uniform(-0.2,0.2);
    if (r==0) matrixMul2x2inPlace(c00,c01,c10,c11,1,0,alpha,1); //Slant
    if (r==1) matrixMul2x2inPlace(c00,c01,c10,c11,1,alpha,0,1); //Slant other way
    if (r==2) matrixMul2x2inPlace(c00,c01,c10,c11,cos(alpha),-sin(alpha),sin(alpha),cos(alpha)); //Rotate
    transformImage(pic->mat, backgroundColor, c00, c01, c10, c11);
    pic->jiggle(rng,16);
  }
  return pic;
}

class CNN : public SparseConvNet {
public:
  CNN (int dimension, int l, int k, ActivationFunction fn, int nInputFeatures, int nClasses, float p=0.0f, int cudaDevice=-1, int nTop=1);
};
CNN::CNN
(int dimension, int l, int k, ActivationFunction fn,
 int nInputFeatures, int nClasses, float p, int cudaDevice, int nTop)
  : SparseConvNet(dimension,nInputFeatures, nClasses, cudaDevice, nTop) {
  for (int i=0;i<=l;i++) {
    addLeNetLayerMP((i+1)*k,2,1,1,1,fn,p*i*1.0f/l);
    addLeNetLayerMP((i+1)*k,2,1,(i<l)?3:1,(i<l)?2:1,fn,p*i*1.0f/l);
  }
  addSoftmaxLayer();
}

int main() {
  std::string baseName="weights/cifar10";

  SpatiallySparseDataset trainSet=Cifar10TrainSet();
  SpatiallySparseDataset testSet=Cifar10TestSet();

  trainSet.summary();
  testSet.summary();
  CNN cnn(2,5,32,VLEAKYRELU,trainSet.nFeatures,trainSet.nClasses,0.0f,cudaDevice);
  //DeepCNet cnn(2,5,32,VLEAKYRELU,trainSet.nFeatures,trainSet.nClasses,0.0f,cudaDevice);

  if (epoch>0)
    cnn.loadWeights(baseName,epoch);
  for (epoch++;;epoch++) {
    std::cout <<"epoch: " << epoch << " " << std::flush;
    cnn.processDataset(trainSet, batchSize,0.003*exp(-0.005 * epoch));
    if (epoch%50==0) {
      cnn.saveWeights(baseName,epoch);
      cnn.processDataset(testSet,  batchSize);
    }
  }
}
