#include "NetworkArchitectures.h"
#include "SpatiallySparseDatasetCIFAR10.h"

int epoch = 0;
int cudaDevice = -1; // PCI bus ID, -1 for default GPU
int batchSize = 50;

Picture *OpenCVPicture::distort(RNG &rng, batchType type) {
  OpenCVPicture *pic = new OpenCVPicture(*this);

  if (epoch <= 400 and type == TRAINBATCH) {
    pic->colorDistortion(rng, 25, 0.15, 2.4, 2.4);
    // 2x2 identity matrix:
    // Generate an affine distortion matrix
    float c00 = 1, c01 = 0, c10 = 0, c11 = 1;
    c00 *= 1 + rng.uniform(-0.2, 0.2); // x stretch
    c11 *= 1 + rng.uniform(-0.2, 0.2); // y stretch
    if (rng.randint(2) == 0)           // Horizontal flip
      c00 *= -1;
    int r = rng.randint(3);
    float alpha = rng.uniform(-0.2, 0.2);
    if (r == 0) // Slant
      matrixMul2x2inPlace(c00, c01, c10, c11, 1, 0, alpha, 1);
    if (r == 1) // Slant
      matrixMul2x2inPlace(c00, c01, c10, c11, 1, alpha, 0, 1);
    if (r == 2) // Rotate
      matrixMul2x2inPlace(c00, c01, c10, c11, cos(alpha), -sin(alpha),
                          sin(alpha), cos(alpha));
    pic->affineTransform(c00, c01, c10, c11);
    pic->jiggle(rng, 16);
  }
  return pic;
}

int main() {
  std::string baseName = "weights/cifar10";

  SpatiallySparseDataset trainSet = Cifar10TrainSet();
  SpatiallySparseDataset testSet = Cifar10TestSet();

  trainSet.summary();
  testSet.summary();
  POFMPSparseConvNet cnn(
      2, 11, 32 /* 32n units in the n-th hidden layer*/, powf(2, 0.3333), PRELU,
      trainSet.nFeatures, trainSet.nClasses,
      0.0f /*dropout multiplier in the range [0,0.5] */, cudaDevice);
  if (epoch > 0)
    cnn.loadWeights(baseName, epoch);
  for (epoch++; epoch <= 410; epoch++) {
    std::cout << "epoch: " << epoch << " " << std::flush;
    cnn.processDataset(trainSet, batchSize, 3e-3 * exp(-epoch * 1e-2), 0.99);
    if (epoch % 10 == 0)
      cnn.saveWeights(baseName, epoch);
    if (epoch % 100 == 0)
      cnn.processDatasetRepeatTest(testSet, batchSize / 2, 12);
  }
  cnn.processDatasetRepeatTest(testSet, batchSize / 2, 100);
}
