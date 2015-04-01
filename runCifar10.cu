int epoch=0;
int cudaDevice=-1; //PCI bus ID, -1 for default GPU
int batchSize=50;

#include "SparseConvNet.h"
#include "OpenCVPicture.h"
#include "readCIFAR10.h"

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

float dropoutProbabilityMultiplier=0;// Set to 0.5 say to use dropout
int nFeaturesPerLevel(int i) {
  return 32*(i+1); //This can be increased
}

class SparseConvNet : public SpatiallySparseCNN {
public:
  SparseConvNet(int nInputFeatures, int nClasses, int cudaDevice,int nTop=1) : SpatiallySparseCNN(nInputFeatures, nClasses, cudaDevice) {
    const float fmpShrink=powf(2,0.3333); //Fractional Max-Pooling ratio.
    int l=12; //Number of levels of FMP
    cnn.push_back(new ColorShiftLayer(0.2,0.3,0.6,0.6));
    for (int i=0;i<l;i++) {
      addLeNetLayerPOFMP(nFeaturesPerLevel(i),2,1,fmpShrink,VLEAKYRELU,dropoutProbabilityMultiplier*i/(l+1));
    }
    addLeNetLayerMP(nFeaturesPerLevel(l),2,1,1,1,VLEAKYRELU,dropoutProbabilityMultiplier*l/(l+1));
    addNetworkInNetworkLayer(nFeaturesPerLevel(l+1),VLEAKYRELU,dropoutProbabilityMultiplier);
    addSoftmaxLayer();
  }
};

int main() {
  string baseName="weights/cifar10";

  SpatialDataset trainSet=Cifar10TrainSet();
  SpatialDataset testSet=Cifar10TestSet();
  trainSet.summary();
  testSet.summary();
  SparseConvNet cnn(trainSet.nFeatures,trainSet.nClasses,cudaDevice);

  if (epoch>0)
    cnn.loadWeights(baseName,epoch);
  for (epoch++;;epoch++) {
    cout <<"epoch: " << epoch << flush;
    cnn.processDataset(trainSet, batchSize,0.003*exp(-0.02 * epoch)); //reduce annealing rate for better results ...
    if (epoch%50==0) {
      cnn.saveWeights(baseName,epoch);
      cnn.processDataset(testSet,  batchSize);
      cnn.processDatasetRepeatTest(testSet, batchSize, 12);
    }
  }
}
