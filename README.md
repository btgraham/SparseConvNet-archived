# SparseConvNet
A spatially sparse convolutoinal neural network
Benjamin Graham, University of Warwick, 2013-2015
GPLv3
Available from
http://www2.warwick.ac.uk/fac/sci/statistics/staff/academic-research/graham/
Requirements: a Nvidia GPU, CUDA sm_20, OpenCV, Boost)
A convolutional neural network that process sparse input images more efficiently.

If you use this software please:
1. Tell me what you are using it for (b.graham@warwick.ac.uk).
2. Cite "Spatially-sparse convolutional neural networks,
         Benjamin Graham,
         http://arxiv.org/abs/1409.6070"

To test SparseConvNet:
- MNIST   -  Put the four "ubyte" files from
             http://yann.lecun.com/exdb/mnist/ into MNIST/ then "run runMNIST.cu"
- CIFAR10 -  Put the binary files from
             http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
             into CIFAR10/ then "run runCIFAR10.cu"

For Plankton images:
- Unzip the competition data into Data/kagglePlankton/
- Run "bashScriptToNormalizeImagesAndGenerateFileLists" in "Data/" which will populate "Data/__kagglePlankton/" (needs the ImageMagick convert utility)
- Run one of
  - "run runKagglePlanktonQuick.cu"  (Fast, not very accurate)
  - "run runKagglePlankton1.cu"      (Slower, more accurate)
  - "run runKagglePlankton2.cu"      (Even slower, more accurate in the end)
  - "run runKagglePlankton3.cu"
- Remove the line "define VALIDATION" in the run...cu files to train the network without a 10% validation set for maximum accuracy.
- If necessary, modify the batchSize which is also set at the top of the .cu files (bigger=faster, smaller=less likely to run out of GPU memory).

References/further reading:

* LeNet-5 convolutional neural networks
  http://yann.lecun.com/exdb/lenet/

* Multi-column deep neural networks for image classification
  Ciresan, Meier and Schmidhuber

* Network In Network;
  Lin, Chen and Yan

* Very Deep Convolutional Networks for Large-Scale Image Recognition
  Simonyan and Zisserman

* SparseConvNet - a convolutional neural network library (GPL3)
  http://www2.warwick.ac.uk/fac/sci/statistics/staff/academic-research/graham/

* Spatially-sparse convolutional neural networks
  Benjamin Graham
  http://arxiv.org/abs/1409.6070

* Fractional Max-Pooling
  Benjamin Graham
  http://arxiv.org/abs/1412.6071

*****************************************************************************
* SparseConvNet is free software: you can redistribute it and/or modify     *
* it under the terms of the GNU General Public License as published by      *
* the Free Software Foundation, either version 3 of the License, or         *
* (at your option) any later version.                                       *
*                                                                           *
* SparseConvNet is distributed in the hope that it will be useful,          *
* but WITHOUT ANY WARRANTY; without even the implied warranty of            *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             *
* GNU General Public License for more details.                              *
*                                                                           *
* You should have received a copy of the GNU General Public License         *
* along with SparseConvNet.  If not, see <http://www.gnu.org/licenses/>.    *
*****************************************************************************
