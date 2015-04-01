SpatialDataset ImageNet2012TrainSet(int scale=256,int n=10000) {
  SpatialDataset dataset;
  dataset.name="ImageNet2012 train set";
  dataset.type=TRAINBATCH;
  dataset.nFeatures=3;
  dataset.nClasses=1000;

  string imageFile;
  int cl;
  ifstream file("Data/imagenet2012/trainingData.txt");
  vector<int> count(1000);
  int nWidth, nHeight, nBBoxx, nBBoxX, nBBoxy, nBBoxY;
  while (file >> cl >> imageFile >> nWidth >> nHeight >> nBBoxx >> nBBoxX >> nBBoxy >> nBBoxY) {
    cl--;
    if (count[cl]++<n) {
      OpenCVPicture*  pic = new OpenCVPicture(string("Data/imagenet2012/")+imageFile,scale,128,cl);
      dataset.pictures.push_back(pic);
    }
  }
  return dataset;
}
SpatialDataset ImageNet2012ValidationSet(int scale=256) {
  SpatialDataset dataset;
  dataset.name="ImageNet2012 validation set";
  dataset.type=TESTBATCH;
  dataset.nFeatures=3;
  dataset.nClasses=1000;

  string imageFile;
  int cl;
  ifstream file("Data/imagenet2012/validationData.txt");
  int nWidth, nHeight, nBBoxx, nBBoxX, nBBoxy, nBBoxY;
  while (file >> cl >> imageFile >> nWidth >> nHeight >> nBBoxx >> nBBoxX >> nBBoxy >> nBBoxY) {
    cl--;
    OpenCVPicture*  pic = new OpenCVPicture(string("Data/imagenet2012/")+imageFile,scale,128,cl);
    dataset.pictures.push_back(pic);
  }
  return dataset;
}
SpatialDataset ImageNet2012TestSet(int scale=256) {
  SpatialDataset dataset;
  dataset.name="ImageNet2012 train set";
  dataset.type=UNLABELLEDBATCH;
  dataset.nFeatures=3;
  dataset.nClasses=1000;

  string imageFile;
  int cl;
  ifstream file("Data/imagenet2012/testData.txt");
  int nWidth, nHeight, nBBoxx, nBBoxX, nBBoxy, nBBoxY;
  while (file >> cl >> imageFile >> nWidth >> nHeight >> nBBoxx >> nBBoxX >> nBBoxy >> nBBoxY) {
    OpenCVPicture*  pic = new OpenCVPicture(string("Data/imagenet2012/")+imageFile,scale,128,-1);
    dataset.pictures.push_back(pic);
  }
  return dataset;
}
