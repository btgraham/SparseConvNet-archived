#include "NetworkArchitectures.h"
#include "SpatiallySparseDatasetKaggleDiabeticRetinopathy.h"
#include <iostream>
#include <string>

int epoch=0;
int cudaDevice=-1; //PCI bus ID, -1 for default GPU
int batchSize=1;
std::string dirName("Data/kaggleDiabeticRetinopathy/500_train/");
std::string dirNameTest("Data/kaggleDiabeticRetinopathy/500_test/");

Picture* OpenCVPicture::distort(RNG& rng, batchType type) {
  OpenCVPicture* pic=new OpenCVPicture(*this);
  pic->loadDataWithoutScaling();
  float
    c00=1, c01=0,  //2x2 identity matrix---starting point for calculating affine distortion matrix
    c10=0, c11=1;
  float r, alpha, beta;
  if (type==TRAINBATCH) {
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
  pic->jiggle(rng,30);
  return pic;
}

class Imagenet : public SparseConvNet {
public:
  Imagenet (int dimension, ActivationFunction fn, int nInputFeatures, int nClasses, int cudaDevice=-1, int nTop=1);
};
Imagenet::Imagenet
(int dimension, ActivationFunction fn,
 int nInputFeatures, int nClasses, int cudaDevice, int nTop)
  : SparseConvNet(dimension,nInputFeatures, nClasses, cudaDevice, nTop) {
  for (int i=1;i<=7;i++) {
    addLeNetLayerMP(32*i,3,1,1,1,fn,0.0f);
    addLeNetLayerMP(32*i,3,1,3,2,fn,0.0f);
  }
  addLeNetLayerMP(32* 9,2,1,1,1,fn);
  addLeNetLayerMP(32* 9,2,1,1,1,fn);
  addLeNetLayerMP(32*10,1,1,1,1,fn);
  addSoftmaxLayer();
}

int main() {
  std::string baseName="Data/kaggleDiabeticRetinopathy/kaggleDiabeticRetinopathyCompetitionModelFiles/kaggleDiabeticRetinopathy3";
  SpatiallySparseDataset trainSet=KDRTrainSet(dirName);
  SpatiallySparseDataset validationSet=KDRValidationSet(dirName);
  SpatiallySparseDataset testSet=KDRTestSet(dirNameTest);
  trainSet.summary();
  validationSet.summary();
  testSet.summary();
  {
    Imagenet cnn(2,VLEAKYRELU,trainSet.nFeatures,trainSet.nClasses,cudaDevice);

    if (epoch==0) {
      SpatiallySparseDataset trainSubset=trainSet.subset(20*batchSize);
      trainSubset.type=RESCALEBATCH;
      cnn.processDataset(trainSubset,batchSize);
    } else {
      cnn.loadWeights(baseName,epoch);
    }
    for (epoch++;epoch<=46;epoch++) {
      std::cout <<"epoch: " << epoch << std::endl;
      for (int i=0;i<3;++i) {
        SpatiallySparseDataset trainSubset=trainSet.subset(12000);
        cnn.processDataset(trainSubset, batchSize,0.003*exp(-epoch*0.05),0.999);
        cnn.saveWeights(baseName,epoch);
      }
      cnn.processDatasetRepeatTest(validationSet, batchSize,1,"kaggleDiabetes3.predictions","","confusionMatrix3.train"); //Monitor progress during training
    }
    cnn.processDatasetRepeatTest(validationSet, batchSize,6,"Data/kaggleDiabeticRetinopathy/kaggleDiabeticRetinopathyCompetitionModelFiles/kaggleDiabetes3_epoch46.validation","","confusionMatrix3.validation");
    SpatiallySparseDataset trainSetAsTestSet=KDRTrainSet(dirName);trainSetAsTestSet.type=TESTBATCH;
    cnn.processDatasetRepeatTest(trainSetAsTestSet, batchSize,6,"Data/kaggleDiabeticRetinopathy/kaggleDiabeticRetinopathyCompetitionModelFiles/kaggleDiabetes3_epoch46.train","","confusionMatrix3.train");
    cnn.processDatasetRepeatTest(testSet, batchSize,6,"Data/kaggleDiabeticRetinopathy/kaggleDiabeticRetinopathyCompetitionModelFiles/kaggleDiabetes3_epoch46.test");
  }
}
