SparseConvNet - Spatially-sparse convolutional networks
=======================================================
Benjamin Graham, University of Warwick, 2013-2015, GPLv3
--------------------------------------------------------

SparseConvNet is a convolutional neural network for processing of sparse data on a variety of lattices dimensional data.
![lattice](/figures/lattices.png)


Implement CNNs on the:
2D) square or triangular lattices
3D) cubic or tetrahedral lattices
4D) hypercubic or hypertetrahedral lattices
![lattice](/figures/shrec.png)
![lattice](/figures/trefoil.png)

Sparsity may be useful in 2d [(i.e. for online handwritting recognition: Spatially-sparse convolutional neural networks)](http://arxiv.org/abs/1409.6070) and even more useful in 3+ dimensions [Sparse 3D convolutional neural networks](http://arxiv.org/abs/1505.02890).

If you use this software please:
1. Tell me what you are using it for (b.graham@warwick.ac.uk).
2. Cite "Spatially-sparse convolutional neural networks,
         Benjamin Graham,
                  http://arxiv.org/abs/1409.6070"

                  To test the CNN:
                  Put the CIFAR-10 data files (http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz) in the
                  Data/CIFAR10/ folder then execute "make cifar10 && cifar10"

                  Dependencies:
                  An Nvidia CUDA sm_20 capable graphics card
                  The CUDA SDK (https://developer.nvidia.com/cuda-downloads)
                  Google sparsehash library (https://code.google.com/p/sparsehash/downloads/list)
                  Armadillo library (http://arma.sourceforge.net/)

                  i.e.
                  sudo apt-get install libarmadillo-dev
                  wget https://sparsehash.googlecode.com/files/sparsehash_2.0.2-1_amd64.deb
                  sudo dpkg -i sparsehash_2.0.2-1_amd64.deb

                  *****************************************************************************
                  SparseConvNet is free software: you can redistribute it and/or modify
                  it under the terms of the GNU General Public License as published by
                  the Free Software Foundation, either version 3 of the License, or
                  (at your option) any later version.

                  SparseConvNet is distributed in the hope that it will be useful,
                  but WITHOUT ANY WARRANTY; without even the implied warranty of
                  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
                  GNU General Public License for more details.

                  You should have received a copy of the GNU General Public License
                  along with SparseConvNet.  If not, see <http://www.gnu.org/licenses/>.
                  **************************************************************************
