#include "SparseConvNet.h"
#include "SpatiallySparseDatasetKagglePlankton.h"

int epoch=0;
int cudaDevice=-1; // PCI bus ID: -1 for default GPU
int batchSize=50; // Increase/decrease according to GPU memory

Picture* OpenCVPicture::distort(RNG& rng, batchType type) {
  OpenCVPicture* pic=new OpenCVPicture(*this);
  pic->loadDataWithoutScaling();
  float
    c00=1, c01=0,  //2x2 identity matrix---starting point for calculating affine distortion matrix
    c10=0, c11=1;
  float r, alpha, beta;
  if (false and type==TRAINBATCH) {
    r=rng.uniform(-0.1,0.1);
    alpha=rng.uniform(0,2*3.1415926535);
    beta=rng.uniform(-0.2,0.2)+alpha;
  } else {
    r=0;
    alpha=rng.uniform(0,2*3.1415926535);
    beta=alpha;
  }
  c00=(1+r)*cos(alpha); c01=(1+r)*sin(alpha);
  c10=-(1-r)*sin(beta); c11=(1-r)*cos(beta);
  if (rng.randint(2)==0) {c00*=-1; c01*=-1;}//Horizontal flip
  pic->affineTransform(c00, c01, c10, c11);
  pic->jiggle(rng,300);
  return pic;
}


int f(int i) {
  return 32*(i+1);
}

class FractionalSparseConvNet : public SparseConvNet {
public:
  FractionalSparseConvNet(int nInputFeatures, int nClasses, int cudaDevice) : SparseConvNet(2,nInputFeatures, nClasses, cudaDevice) {
    int l=12;
    float p=0.0f;
    const float fmpShrink=1.414;
    for (int i=0;i<l;i++) {
      addLeNetLayerPOFMP(f(i),2,1,2,fmpShrink,VLEAKYRELU,p*std::max(i-4,0)/(l-3));
    }
    addLeNetLayerPOFMP(f(l),2,1,1,1,VLEAKYRELU,p*std::max(l-4,0)/(l-3));
    addTerminalPoolingLayer(32);
    addLeNetLayerPOFMP(f(l+1),1,1,1,1,VLEAKYRELU,p*std::max(l-3,0)/(l-3));
    addSoftmaxLayer();
  }
};


int main() {
  std::string baseName="weights/plankton";

  KagglePlanktonLabeledDataSet trainSet("Data/kagglePlankton/classList","Data/kagglePlankton/train/",TRAINBATCH,255);
  trainSet.summary();
  KagglePlanktonLabeledDataSet cheekyExtraTrainSet("Data/kagglePlankton/classList","Data/kagglePlankton/testPrivate/",TRAINBATCH,255);  //Use the "private test set" as extra training data.
  cheekyExtraTrainSet.summary();
  KagglePlanktonLabeledDataSet valSet("Data/kagglePlankton/classList","Data/kagglePlankton/testPublic/",TESTBATCH,255);
  valSet.summary();

  FractionalSparseConvNet cnn(trainSet.nFeatures,trainSet.nClasses,cudaDevice);

  if (epoch>0) {
    cnn.loadWeights(baseName,epoch);
    cnn.processDatasetRepeatTest(valSet, batchSize/2, 12);
  }
  for (epoch++;;epoch++) {
    std::cout <<"epoch: " << epoch << std::endl;
    float lr=0.003*exp(-0.1*epoch);
    for (int i=0;i<3;++i) {
      cnn.processDataset(trainSet, batchSize,lr,0.999);
      cnn.saveWeights(baseName,epoch);
      cnn.processDataset(cheekyExtraTrainSet, batchSize,lr,0.999);
      cnn.saveWeights(baseName,epoch);
    }
    cnn.processDatasetRepeatTest(valSet, batchSize, 1);
  }

  // For unlabelled data (but there is overlap between this "test" data and our expanded training set!!!)
  // KagglePlanktonUnlabeledDataSet testSet("Data/kagglePlankton/classList","Data/kagglePlankton/test/",255);
  // testSet.summary();
  // cnn.processDatasetRepeatTest(testSet, batchSize/2, 24,"plankton.predictions",testSet.header);
}
