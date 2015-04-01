//Pick a random rotation in color space and apply it to the training data to force learning of shape not color.

#include "SparseConvNet.h"
#include "OpenCVPicture.h"
#include "OpenCVPicture_AffineTransform.h"
//Picture* OpenCVPicture::distort(RNG& rng, batchType type) {
//  OpenCVPicture* pic=new OpenCVPicture(*this);
//  pic->loadData();
//  if (type==TRAINBATCH) {
//    pic->jiggle(rng,2);
//  }
//  return pic;
//}
#include "readCIFAR10.h"

int epoch=0;
int cudaDevice=-1;

class SparseConvNet : public SpatiallySparseCNN {
public:
  SparseConvNet(int nInputFeatures, int nClasses, int cudaDevice) : SpatiallySparseCNN(nInputFeatures, nClasses, cudaDevice) {
    cnn.push_back(new ColorShiftLayer(0.2,0.2,0.6,0.6));
    int i=1;
    float alpha=powf(2,0.5);
    addLeNetLayerROFMP(32*i++,2,1,alpha,VLEAKYRELU,0);
    addLeNetLayerROFMP(32*i++,2,1,alpha,VLEAKYRELU,0);
    addLeNetLayerROFMP(32*i++,2,1,alpha,VLEAKYRELU,0);
    addLeNetLayerROFMP(32*i++,2,1,alpha,VLEAKYRELU,0);
    addLeNetLayerROFMP(32*i++,2,1,alpha,VLEAKYRELU,0);
    addLeNetLayerROFMP(32*i++,2,1,alpha,VLEAKYRELU,0);
    addLeNetLayerROFMP(32*i++,2,1,alpha,VLEAKYRELU,0);
    addLeNetLayerROFMP(32*i++,2,1,alpha,VLEAKYRELU,0);
    addLeNetLayerROFMP(32*i++,2,1,alpha,VLEAKYRELU,0);
    addLeNetLayerMP(32*i++,2,1,1,1,VLEAKYRELU,0);
    addIndexLearnerLayer();
  }
};


int main() {
  string baseName="weights/Cifar10IndexLearner";

  SpatialDataset trainSet=Cifar10TrainSet();   //For training
  SpatialDataset trainSet2=Cifar10TrainSet();trainSet2.type=TESTBATCH; //For calculating the top hidden layer for the training set (without dropout or data augmentation)
  SpatialDataset testSet=Cifar10TestSet(); //For calculating the top hidden layer for the test set

  int batchSize=100;
  trainSet.summary();
  trainSet2.summary();
  testSet.summary();
  SparseConvNet cnn(trainSet.nFeatures,trainSet.pictures.size(),cudaDevice);
  if (epoch>0) {
    cnn.loadWeights(baseName,epoch);
    cnn.processIndexLearnerDataset(trainSet2, batchSize,0);
    cnn.processIndexLearnerDataset(testSet, batchSize,0);
  }
  for (epoch++;epoch<=100000;epoch++) {
    cout <<"epoch: " << epoch << endl;
    cnn.processIndexLearnerDataset(trainSet, batchSize,0.0003);
    if (epoch%10==0) {
      cnn.saveWeights(baseName,epoch);
    }
  }
  {//After pretraining, convert to a regular neural network
    cnn.nOutputFeatures=dynamic_cast<IndexLearnerLayer*>(cnn.cnn.back())->nFeaturesIn;
    delete cnn.cnn.back();
    cnn.cnn.pop_back();
    cnn.nClasses=trainSet.nClasses;
    cnn.addLearntLayer(1024);cout <<endl;
    cnn.addLearntLayer(1024);cout <<endl;
    cnn.addSoftmaxLayer();cout <<endl;
  }
  for (;;epoch++) {
    cout <<"epoch: " << epoch << flush;
    cnn.processDataset(trainSet, batchSize,0.0003);
    if (epoch%10==0) {
      cnn.saveWeights(baseName,epoch);
      cnn.processDatasetRepeatTest(testSet, batchSize, 1);
    }
  }
}
