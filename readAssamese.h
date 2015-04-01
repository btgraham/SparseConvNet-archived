// Download and extract the archive:
// wget  http://archive.ics.uci.edu/ml/machine-learning-databases/00208/Online%20Handwritten%20Assamese%20Characters%20Dataset.rar
// tar x "Online Handwritten Assamese Characters Dataset.rar"
// Run a command to standardize the file extensions:
// for x in W*/*; do echo $x; rename 's/TXT/txt/' $x; done

#define ASSAMESETRAININGSAMPLES 36
void readAssameseWriterDirectory(int writerNumber, vector<Picture*> &characters) {
  char filenametemplate[]= "Data/Online Handwritten Assamese Characters Dataset/W%d/%d.%d.txt";
  char filename[100];
  for (int characterNumber=0;characterNumber<183;characterNumber++) {
    sprintf(filename, filenametemplate, writerNumber, characterNumber+1,writerNumber);
    ifstream file(filename);
    if (!file) {
      cout << "Cannot find " << filename << endl;
      cout << "Please download it from the UCI Machine Learning Repository:" << endl;
      cout << "  http://archive.ics.uci.edu/ml/datasets/Online+Handwritten+Assamese+Characters+Dataset" << endl;
      cout << "and run a command to standardize the file extensions, i.e." << endl;
      cout << "for x in W*/*; do echo $x; rename 's/TXT/txt/' $x; done" << endl;
      exit(EXIT_FAILURE);}
    string str;
    int numberOfStrokes;
    getline(file,str);
    file >> str >>numberOfStrokes;
    getline(file,str);
    getline(file,str);
    OnlinePicture* character = new OnlinePicture(characterNumber);
    character->ops.resize(numberOfStrokes);
    while(!file.eof()) {
      unsigned char peek=file.peek();
      if (peek>=' ' && peek<='9') {
        FloatPoint p;
        int zzz,strokeNumber;
        file >> p.y >> p.x >> zzz >> strokeNumber;
        character->ops[strokeNumber-1].push_back(p);
      } else getline(file,str);
    }
    normalize(character->ops);
    characters.push_back(character);
  }
}

SpatialDataset AssameseTrainSet() {
  SpatialDataset dataset;
  dataset.name="Assamese train set--36 writers";
  dataset.type=TRAINBATCH;
  dataset.nFeatures=nInputFeatures;
  dataset.nClasses=183;
  for (int writer=1;writer<=ASSAMESETRAININGSAMPLES;writer++)
    readAssameseWriterDirectory(writer,dataset.pictures);
  calculateRegularizingConstants(dataset);
  return dataset;
}
SpatialDataset AssameseTestSet() {
  SpatialDataset dataset;
  dataset.name="Assamese test set--9 writers";
  dataset.type=TESTBATCH;
  dataset.nFeatures=nInputFeatures;
  dataset.nClasses=183;
  for (int writer=ASSAMESETRAININGSAMPLES+1;writer<=45;writer++)
    readAssameseWriterDirectory(writer,dataset.pictures);
  return dataset;
}
