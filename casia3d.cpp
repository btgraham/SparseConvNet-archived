#include "SparseConvNet.h"
#include "SpatiallySparseDatasetCasiaOLHWDB.h"
#include "OnlineHandwritingPicture.h"

int epoch = 0;
int cudaDevice = -1; // PCI bus ID, -1 for default GPU
int batchSize = 50;

Picture *OnlineHandwritingPicture::distort(RNG &rng, batchType type) {
  OnlineHandwritingPicture *pic = new OnlineHandwritingPicture(*this);
  if (type == TRAINBATCH) {
    arma::mat aff = arma::eye(2, 2);
    aff(0, 0) += rng.uniform(-0.3, 0.3); // x stretch
    aff(1, 1) += rng.uniform(-0.3, 0.3); // y stretch
    int r = rng.randint(3);
    float alpha = rng.uniform(-0.3, 0.3);
    arma::mat x = arma::eye(2, 2);
    if (r == 0)
      x(0, 1) = alpha;
    if (r == 1)
      x(1, 0) = alpha;
    if (r == 2) {
      x(0, 0) = cos(alpha);
      x(0, 1) = -sin(alpha);
      x(1, 0) = sin(alpha);
      x(1, 1) = cos(alpha);
    }
    aff = aff * x;
    arma::mat y(1, 2);
    y(0, 0) = renderSize * rng.uniform(-0.1875, 0.1875);
    y(0, 1) = renderSize * rng.uniform(-0.1875, 0.1875);
    for (int i = 0; i < ops.size(); ++i)
      pic->ops[i] = pic->ops[i] * aff + arma::repmat(y, pic->ops[i].n_rows, 1);
    pic->offset3d = rng.uniform(-0.1, 0.1);
  }
  return pic;
}

class CNN : public SparseConvNet {
public:
  CNN(int dimension, int l, int k, ActivationFunction fn, int nInputFeatures,
      int nClasses, float p = 0.0f, int cudaDevice = -1, int nTop = 1)
      : SparseConvNet(dimension, nInputFeatures, nClasses, cudaDevice, nTop) {
    for (int i = 0; i <= l; i++)
      addLeNetLayerMP((i + 1) * k * 0 + 32 << i, 2 + (i == 0) + (i == l), 1,
                      (i < l) ? 3 : 1, (i < l) ? 2 : 1, fn, p * i * 1.0f / l);
    addSoftmaxLayer();
  }
};

int main() {
  std::string baseName = "weights/casia3d";

  SpatiallySparseDataset trainSet = CasiaOLHWDB11TrainSet(40, SpaceTime3d);
  SpatiallySparseDataset testSet = CasiaOLHWDB11TestSet(40, SpaceTime3d);

  trainSet.summary();
  testSet.summary();
  CNN cnn(3, 4, 64, RELU, trainSet.nFeatures, trainSet.nClasses, 0.0f,
          cudaDevice);
  // DeepC2
  // cnn(3,4,64,RELU,trainSet.nFeatures,trainSet.nClasses,0.0f,cudaDevice);
  cnn.calculateInputRegularizingConstants(trainSet);

  if (epoch > 0)
    cnn.loadWeights(baseName, epoch);
  for (epoch++;; epoch++) {
    std::cout << "epoch:" << epoch << ": " << std::flush;
    cnn.processDataset(trainSet, batchSize, 0.003 * exp(-0.03 * epoch), 0.995);
    cnn.saveWeights(baseName, epoch);
    if (epoch % 10 == 0) {
      cnn.processDataset(testSet, batchSize);
    }
  }
}
