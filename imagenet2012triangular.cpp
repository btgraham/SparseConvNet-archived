#include "SparseConvNet.h"
#include "SpatiallySparseDatasetImageNet2012.h"
#include <iostream>
#include <string>

int epoch=0;
int cudaDevice=-1; //PCI bus ID, -1 for default GPU
int batchSize=4;

Picture* OpenCVPicture::distort(RNG& rng, batchType type) {
  OpenCVPicture* pic=new OpenCVPicture(*this);
  if (type==TRAINBATCH)
    pic->loadData(224+rng.randint(224));
  else
    pic->loadData(336);
  float
    c00=1, c01=0,  //2x2 identity matrix---starting point for calculating affine distortion matrix
    c10=0, c11=1;
  if (type==TRAINBATCH)  {
    float r=rng.uniform(-0.1,0.1);
    float alpha=rng.uniform(-0.3,0.3);
    float beta=rng.uniform(-0.2,0.2)+alpha;
    c00=(1+r)*cos(alpha); c01=(1+r)*sin(alpha);
    c10=-(1-r)*sin(beta); c11=(1-r)*cos(beta);
  }
  if (rng.randint(2)==0) {c00*=-1; c01*=-1;}//Horizontal flip
  matrixMul2x2inPlace(c00,         c01,
                      c10,         c11,
                      1,           0,
                      -powf(3,-0.5),powf(0.75,-0.5)); //to triangular lattice
  pic->affineTransform(c00, c01, c10, c11);
  pic->jiggleFit(rng,224);
  return pic;
}

class ImagenetTriangular : public SparseConvTriangLeNet {
public:
  ImagenetTriangular (int dimension, ActivationFunction fn, int nInputFeatures, int nClasses, int cudaDevice=-1, int nTop=1);
};
ImagenetTriangular::ImagenetTriangular
(int dimension, ActivationFunction fn,
 int nInputFeatures, int nClasses, int cudaDevice, int nTop)
  : SparseConvTriangLeNet(dimension,nInputFeatures, nClasses, cudaDevice, nTop) {
  addLeNetLayerMP( 64,7,2,2,2,fn,0.0f,10);
  addLeNetLayerMP(128,3,1,3,2,fn,0.0f,4);
  addLeNetLayerMP(256,3,1,3,2,fn,0.0f,4);
  addLeNetLayerMP(384,3,1,3,2,fn,0.0f,4);
  addLeNetLayerMP(512,3,1,1,1,fn,0.0f,4);
  addTerminalPoolingLayer(32);
  addLeNetLayerMP(1024,1,1,1,1,fn);
  addSoftmaxLayer();
}


int main() {
  std::string baseName="weights/imagenet2012triangular";

  SpatiallySparseDataset trainSet=ImageNet2012TrainSet();
  SpatiallySparseDataset validationSet=ImageNet2012ValidationSet();
  SpatiallySparseDataset validationSubset=validationSet.subset(500);
  trainSet.summary();
  validationSubset.summary();

  {
    ImagenetTriangular cnn(2,VLEAKYRELU,trainSet.nFeatures,trainSet.nClasses,cudaDevice,5);

    if (epoch==0) {
      SpatiallySparseDataset trainSubset=trainSet.subset(100*batchSize);
      trainSubset.type=RESCALEBATCH;
      cnn.processDataset(trainSubset,batchSize);
    } else {
      cnn.loadWeights(baseName,epoch);
      cnn.processDatasetRepeatTest(validationSubset, batchSize,3);
    }
    for (epoch++;;epoch++) {
      std::cout <<"epoch: " << epoch << std::endl;
      SpatiallySparseDataset trainSubset=trainSet.subset(32000);
      cnn.processDataset(trainSubset, batchSize,0.003,0.999);
      cnn.saveWeights(baseName,epoch);
      cnn.processDatasetRepeatTest(validationSubset, batchSize,1);
    }
  }
}
