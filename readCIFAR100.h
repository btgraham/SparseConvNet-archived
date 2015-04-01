void readCIFAR100File(vector<Picture*> &characters, const char* filename) {
  ifstream file(filename,ios::in|ios::binary);
  if (!file) {
    cout <<"Cannot find " << filename << endl;
    exit(EXIT_FAILURE);
  }
  unsigned char label[2];
  while (file.read((char*)label,2)) {
    OpenCVPicture* character = new OpenCVPicture(32,32,3,128,label[1]);
    unsigned char bitmap[3072];
    file.read((char*)bitmap,3072);
    for (int x=0;x<32;x++) {
      for (int y=0;y<32;y++) {
        for (int c=0;c<3;c++) {
          character->mat.ptr()[y*96+x*3+(2-c)]=bitmap[c*1024+y*32+x];
        }
      }
    }
    characters.push_back(character);
  }
  file.close();
}
SpatialDataset Cifar100TrainSet() {
  SpatialDataset dataset;
  dataset.name="CIFAR-100 train set";
  dataset.type=TRAINBATCH;
  dataset.nFeatures=3;
  dataset.nClasses=100;
  char filenameTrain[]="Data/CIFAR100/train.bin";
  readCIFAR100File(dataset.pictures,filenameTrain);
  return dataset;
}
SpatialDataset Cifar100TestSet() {
  SpatialDataset dataset;
  dataset.name="CIFAR-100 test set";
  dataset.type=TESTBATCH;
  dataset.nFeatures=3;
  dataset.nClasses=100;
  char filenameTest[]="Data/CIFAR100/test.bin";
  readCIFAR100File(dataset.pictures,filenameTest);
  return dataset;
}
