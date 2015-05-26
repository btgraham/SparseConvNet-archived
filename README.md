# SparseConvNet
## A spatially-sparse convolutional neural network
### [Ben Graham](http://www2.warwick.ac.uk/fac/sci/statistics/staff/academic-research/graham/), [University of Warwick](http://www2.warwick.ac.uk/fac/sci/statistics/), 2013-2015, GPLv3

SparseConvNet is a convolutional neural network for processing sparse data on a variety of lattices, i.e.
(i) the square lattice,
(ii) the triangular lattice,
(iii) the cubic lattice,
(iv) the tetrahedral lattice, ...  
![lattice](/figures/lattices.png)  
... and of course the hyper-cubic and hyper-tetrahedral 4D lattices as well.

Data is sparse if most sites take the value zero. For example, if a loop of string has a knot in it, and you trace the shape of the string in a 3D lattice, most sites will not form part of the knot (left). Applying a 2x2x2 convolution (middle), and a pooling operation (right), the set of non-zero sites stays fairly small:
![lattice](/figures/trefoil.png)

This can be used to analyse 3D models, or space-time paths.
Here are some examples from a [3D object dataset](http://www.itl.nist.gov/iad/vug/sharp/contest/2014/Generic3D/index.html). The insides are hollow, so the data is fairly sparse. The computational complexity of processing the models is related to the [fractal dimension](http://en.wikipedia.org/wiki/Fractal_dimension) of the underlying objects.

![lattice](/figures/shrec.png)
Top row: four exemplars of snakes. Bottom row: an ant, an elephant, a robot and a tortoise.

## [Wiki](https://github.com/btgraham/SparseConvNet/wiki)
## [Dependencies and Installation](https://github.com/btgraham/SparseConvNet/wiki/Installation)

### To test the CNN:
1. Put the [CIFAR-10 .bin data files](http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz) in the Data/CIFAR10/ folder
2. Execute "make cifar10 && ./cifar10"

### Dependencies:
1. An Nvidia CUDA sm_20 capable graphics card and the [CUDA SDK](https://developer.nvidia.com/cuda-downloads)
2. The [OpenCV](http://opencv.org/) library
3. The [Armadillo](http://arma.sourceforge.net/) library
4. Google's [Sparsehash](https://code.google.com/p/sparsehash/downloads/list) library

To install dependencies 2-4 on Ubuntu:
sudo apt-get install libarmadillo-dev libboost-dev libopencv-core-dev libopencv-highgui-dev
wget https://sparsehash.googlecode.com/files/sparsehash_2.0.2-1_amd64.deb
sudo dpkg -i sparsehash_2.0.2-1_amd64.deb

### References
1. [Spatially-sparse convolutional neural networks](http://arxiv.org/abs/1409.6070)
2. [Sparse 3D convolutional neural networks](http://arxiv.org/abs/1505.02890)

**************************************************************************
SparseConvNet is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

SparseConvNet is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
[GNU General Public License](http://www.gnu.org/licenses/) for more details.
**************************************************************************
