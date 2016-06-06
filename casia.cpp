#include "SparseConvNet.h"
#include "NetworkArchitectures.h"
#include "SpatiallySparseDatasetCasiaOLHWDB.h"
#include "OnlineHandwritingPicture.h"

int epoch = 0;
int cudaDevice = -1; // PCI bus ID, -1 for default GPU
int batchSize = 100;

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
    for (unsigned int i = 0; i < ops.size(); ++i)
      pic->ops[i] = pic->ops[i] * aff + arma::repmat(y, pic->ops[i].n_rows, 1);
    pic->offset3d = rng.uniform(-0.1, 0.1);
  }
  return pic;
}

int main() {
  std::string baseName = "weights/casia";
  SpatiallySparseDataset trainSet = CasiaOLHWDB11TrainSet(64, Octogram);
  SpatiallySparseDataset testSet = CasiaOLHWDB11TestSet(64, Octogram);

  trainSet.summary();
  testSet.summary();

  DeepC3C3Valid cnn(2, 5, 96, VLEAKYRELU, trainSet.nFeatures, trainSet.nClasses,
                    0.0f, cudaDevice);

  cnn.calculateInputRegularizingConstants(trainSet);

  if (epoch > 0)
    cnn.loadWeights(baseName, epoch / 100);
  for (epoch++;; epoch++) {
    std::cout << "mini-epoch:" << epoch << ": " << std::flush;
    auto trainSubset = trainSet.subset(50000);
    cnn.processDataset(trainSubset, batchSize, 0.002 * exp(-epoch * 0.0025));
    if (epoch % 18 == 0) {
      cnn.processDataset(testSet, batchSize);
      cnn.saveWeights(baseName, epoch / 100);
    }
  }
}
