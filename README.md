SparseConvNet - Spatially-sparse convolutional networks
=======================================================
Benjamin Graham, University of Warwick, 2013-2015, GPLv3
--------------------------------------------------------

SparseConvNet is a convolutional neural network for processing of sparse data on a variety of lattices dimensional data, i.e.
(i) the square lattice
(ii) The triagular lattice
(iii) The cubic lattice
(iv) The tetrahedral lattice
![lattice](/figures/lattices.png)
and similarly in 4D.

Data is sparse if most sites take the value zero. For example if you trace the shape of a knot, most of space is not visited by the string. Applying convolution, sparsity also hold in the large initial layers:
![lattice](/figures/trefoil.png)

In 3D, this can be used to analyse 3d models. The insides can be hollow, so the data is sparse.
![lattice](/figures/shrec.png)
Examples from the 3D object dataset. Top row: four exemplars of snakes.
Bottom row: an ant, an elephant, a robot and a tortoise.


To test the CNN:
Put the [CIFAR-10 data files](http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz) in the Data/CIFAR10/ folder then execute "make cifar10 && cifar10"

Dependencies:
An Nvidia CUDA sm_20 capable graphics card
The CUDA SDK (https://developer.nvidia.com/cuda-downloads)
Google sparsehash library (https://code.google.com/p/sparsehash/downloads/list)
Armadillo library (http://arma.sourceforge.net/)

i.e.
sudo apt-get install libarmadillo-dev
wget https://sparsehash.googlecode.com/files/sparsehash_2.0.2-1_amd64.deb
sudo dpkg -i sparsehash_2.0.2-1_amd64.deb

**************************************************************************
SparseConvNet is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

SparseConvNet is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
(GNU General Public License)[http://www.gnu.org/licenses/] for more details.
**************************************************************************
