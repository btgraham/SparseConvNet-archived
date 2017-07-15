CC=g++
CFLAGS=--std=c++11 -O3
NVCC=nvcc
NVCCFLAGS=--std=c++11 -arch sm_20 -O3
OBJ=BatchProducer.o ConvolutionalLayer.o ConvolutionalTriangularLayer.o IndexLearnerLayer.o MaxPoolingLayer.o MaxPoolingTriangularLayer.o NetworkArchitectures.o NetworkInNetworkLayer.o NetworkInNetworkPReLULayer.o Picture.o Regions.o Rng.o SigmoidLayer.o SoftmaxClassifier.o SparseConvNet.o SparseConvNetCUDA.o SpatiallySparseBatch.o SpatiallySparseBatchInterface.o SpatiallySparseDataset.o SpatiallySparseLayer.o TerminalPoolingLayer.o readImageToMat.o types.o utilities.o vectorCUDA.o ReallyConvolutionalLayer.o vectorHash.o
OBJCV=$(OBJ) OpenCVPicture.o SpatiallySparseDatasetOpenCV.o
OBJCVT=$(OBJ) OpenCVTriangularPicture.o SpatiallySparseDatasetOpenCV.o
LIBS=-lopencv_core -lopencv_highgui -lopencv_imgproc -lrt -lcublas -larmadillo

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)
%.o: %.cu $(DEPS)
	$(NVCC) -c -o $@ $< $(NVCCFLAGS)

clean:
	rm *.o
casia: $(OBJ) OnlineHandwritingPicture.o SpatiallySparseDatasetCasiaOLHWDB.o casia.o
	$(NVCC) -o casia $(OBJ) OnlineHandwritingPicture.o SpatiallySparseDatasetCasiaOLHWDB.o casia.o $(LIBS) $(NVCCFLAGS)

pendigits: $(OBJ) OnlineHandwritingPicture.o SpatiallySparseDatasetPenDigits.o pendigits.o
	$(NVCC) -o pendigits $(OBJ) OnlineHandwritingPicture.o SpatiallySparseDatasetPenDigits.o pendigits.o $(LIBS) $(NVCCFLAGS)

cifar10: $(OBJCV) SpatiallySparseDatasetCIFAR10.o cifar10.o
	$(NVCC) -o cifar10 $(OBJCV) SpatiallySparseDatasetCIFAR10.o cifar10.o $(LIBS) $(NVCCFLAGS)

cifar10big: $(OBJCV) SpatiallySparseDatasetCIFAR10.o cifar10big.o
	$(NVCC) -o cifar10big $(OBJCV) SpatiallySparseDatasetCIFAR10.o cifar10big.o $(LIBS) $(NVCCFLAGS)

cifar10FullyConnected: $(OBJCV) SpatiallySparseDatasetCIFAR10FullyConnected.o cifar10FullyConnected.o
	$(NVCC) -o cifar10FullyConnected $(OBJCV) SpatiallySparseDatasetCIFAR10FullyConnected.o cifar10FullyConnected.o $(LIBS) $(NVCCFLAGS)

cifar100: $(OBJCV) SpatiallySparseDatasetCIFAR100.o cifar100.o
	$(NVCC) -o cifar100 $(OBJCV) SpatiallySparseDatasetCIFAR100.o cifar100.o $(LIBS) $(NVCCFLAGS)

shrec2015: $(OBJ) Off3DFormatPicture.o SpatiallySparseDatasetSHREC2015.o shrec2015.o
	$(NVCC) -o shrec2015 $(OBJ) Off3DFormatPicture.o SpatiallySparseDatasetSHREC2015.o shrec2015.o $(LIBS) $(NVCCFLAGS)

shrec2015_: $(OBJ) Off3DFormatPicture.o SpatiallySparseDatasetSHREC2015.o shrec2015_.o
	$(NVCC) -o shrec2015_ $(OBJ) Off3DFormatPicture.o SpatiallySparseDatasetSHREC2015.o shrec2015_.o $(LIBS) $(NVCCFLAGS)

casia3d: $(OBJ) OnlineHandwritingPicture.o SpatiallySparseDatasetCasiaOLHWDB.o casia3d.o
	$(NVCC) -o casia3d $(OBJ) OnlineHandwritingPicture.o SpatiallySparseDatasetCasiaOLHWDB.o casia3d.o $(LIBS) $(NVCCFLAGS)

cifar10triangular: $(OBJCVT) SpatiallySparseDatasetCIFAR10.o cifar10triangular.o
	$(NVCC) -o cifar10triangular $(OBJCVT) SpatiallySparseDatasetCIFAR10.o cifar10triangular.o $(LIBS) $(NVCCFLAGS)

shrec2015triangular: $(OBJ) Off3DFormatTriangularPicture.o SpatiallySparseDatasetSHREC2015.o shrec2015triangular.o
	$(NVCC) -o shrec2015triangular $(OBJ) Off3DFormatTriangularPicture.o SpatiallySparseDatasetSHREC2015.o shrec2015triangular.o $(LIBS) $(NVCCFLAGS)

cvap_rha: $(OBJ) CVAP_RHA_Picture.o SpatiallySparseDatasetCVAP_RHA.o cvap_rha.o
	$(NVCC) -o cvap_rha $(OBJ) CVAP_RHA_Picture.o SpatiallySparseDatasetCVAP_RHA.o cvap_rha.o $(LIBS) $(NVCCFLAGS)

ucf101: $(OBJ) UCF101Picture.o SpatiallySparseDatasetUCF101.o ucf101.o
	$(NVCC) -o ucf101 $(OBJ) UCF101Picture.o SpatiallySparseDatasetUCF101.o ucf101.o $(LIBS) $(NVCCFLAGS)

imagenet2012triangular: $(OBJCVT) SpatiallySparseDatasetImageNet2012.o imagenet2012triangular.o
	$(NVCC) -o imagenet2012triangular $(OBJCVT) SpatiallySparseDatasetImageNet2012.o imagenet2012triangular.o $(LIBS) $(NVCCFLAGS)

mnist: $(OBJCV) SpatiallySparseDatasetMnist.o mnist.o
	$(NVCC) -o mnist $(OBJCV) SpatiallySparseDatasetMnist.o mnist.o $(LIBS) $(NVCCFLAGS)

plankton: $(OBJCV) plankton.o
	$(NVCC) -o plankton $(OBJCV) plankton.o $(LIBS) $(NVCCFLAGS)

cifar10indexLearning: $(OBJCV) SpatiallySparseDatasetCIFAR10.o cifar10indexLearning.o
	$(NVCC) -o cifar10indexLearning $(OBJCV) SpatiallySparseDatasetCIFAR10.o cifar10indexLearning.o $(LIBS) $(NVCCFLAGS)
