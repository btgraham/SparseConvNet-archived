void loadSVHN(string filename, vector<Picture*> &characters) {
  ifstream f(filename.c_str());
  if (!f) {
    cout <<"Cannot find " << filename << endl;
    exit(EXIT_FAILURE);}
  int n;
  f.read((char*)&n,4);
  for (int i=0;i<n;i++) {
    unsigned char p[3072], l;
    f.read((char*)&p,3072);
    f.read((char*)&l,1);
    OpenCVPicture* character = new OpenCVPicture(32,32,3,128,l);
    for (int x=0;x<32;x++) {
      for (int y=0;y<32;y++) {
        for (int c=0;c<3;c++) {
          character->mat.ptr()[y*96+x*3+(2-c)]=p[c*1024+y*32+x];
        }
      }
    }
    characters.push_back(character);
  }
}

SpatialDataset SvhnTrainSetAndExtraSet() {
  SpatialDataset dataset;
  dataset.name="SVHN Training (+ Extra) Set";
  dataset.type=TRAINBATCH;
  dataset.nFeatures=3;
  dataset.nClasses=10;
  loadSVHN(string ("Data/svhn/train.data"), dataset.pictures);
  loadSVHN(string ("Data/svhn/extra.data"), dataset.pictures);
  return dataset;
}
SpatialDataset SvhnTestSet() {
  SpatialDataset dataset;
  dataset.name="SVHN Test Set";
  dataset.type=TESTBATCH;
  dataset.nFeatures=3;
  dataset.nClasses=10;
  loadSVHN(string ("Data/svhn/test.data"), dataset.pictures);
  return dataset;
}
