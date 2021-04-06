# prosecco_nmr

<img src="https://github.com/maximosanz/prosecco_nmr/blob/master/Images/prosecco.png" width="700" title="PROSECCO">

# Deep learning prediction of NMR chemical shifts from protein sequence

`prosecco_nmr` is a python package that generates [NMR chemical shift](https://en.wikipedia.org/wiki/Chemical_shift) data from a protein's [primary sequence](https://en.wikipedia.org/wiki/Protein_primary_structure).

Predictions are performed by a 120-layer residual [convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network), trained on a data-set of over 4 million data points from the [biological magnetic resonance data bank](https://bmrb.io/).

The deep learning module of `prosecco_nmr` is interfaced with DeepMind's [AlphaFold](https://github.com/deepmind/deepmind-research/tree/master/alphafold_casp13) algorithm, which is required for prediction.

A simpler neural network model (v1.0) is also included, whose performance is published here:

[Sanz-Hernández, M. & De Simone, A. J Biomol NMR (2017)](https://link.springer.com/article/10.1007%2Fs10858-017-0145-2)

Predictions using the v1 model can be obtained through a web server: http://desimone.bio.ic.ac.uk/prosecco/

- [Dependencies](#dependencies)
- [Neural network architecture](#neural-network-architecture)

## Dependencies

Running the end-to-end `prosecco_nmr` model requires many dependencies, including scientific software for protein bioinformatics:

- Python 3.6+ (recommended installation via [Anaconda](https://www.anaconda.com/))

Python modules:

- [NumPy 1.20](https://numpy.org/)
- [Tensorflow 2.4](https://tensorflow.org/) (GPU compatible)
- [Pandas 1.2](https://pandas.pydata.org/)
- [scikit-learn 0.24](https://scikit-learn.org/stable/index.html)
- [Biopython 1.78](https://biopython.org/)
- [PyNMRSTAR 3.1](https://github.com/uwbmrb/PyNMRSTAR/)

Other software:

- [AlphaFold 1](https://github.com/deepmind/deepmind-research/tree/master/alphafold_casp13)
- [NCBI BLAST+ 2.10](https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Web&PAGE_TYPE=BlastDocs&DOC_TYPE=Download)
- [PSIPRED 4](http://bioinf.cs.ucl.ac.uk/software_downloads/)
- [UCLUST](https://www.drive5.com/usearch/manual/uclust_algo.html)
- [Octave 3.8](https://www.gnu.org/software/octave/index)
- [HH-suite 3](https://github.com/soedinglab/hh-suite)

`prosecco_nmr` can be installed as a python module using pip, by navigating to the root directory and running

` pip install .`

### GPU support

`prosecco_nmr` training and evaluation can be performed using GPUs as long as the GPU-compatible version of tensorflow is intalled.

In order to run the AlphaFold model on GPUs (at the feature generation step), the `contacts.py` file inside `alphafold_patch` must be copied to the `alphafold_casp13` root directory (in the AlphaFold installation).

## Neural network architecture

The core of `prosecco_nmr` is a residual convolutional neural network, that takes in a 2D pairwise sequence crop of 64x64 residues. The features for each residue pair are tiled to form a vector of size `2848` (one-hot encoded sequence information + distogram, torsion, ASA and secondary structure predictions from AlphaFold).

The features are projected down onto `64` channels, and they enter a residual block where a 3x3 dilated convolution is applied. Residual blocks cycle through dilations of different size (1, 2 4 and 8) and include different channel size (512, 256 and 128).

Finally, the output of the last residual block undergoes a final convolution and a `1x6` mean pooling and SoftMax activation, to obtain 6 predictions (one per backbone atom) for each of the 64 residues in the input crop. Predictions are encoded as a probability density function of the secondary chemical shift for each atom.

<img src="https://github.com/maximosanz/prosecco_nmr/blob/using_alphafold/Images/NN_architecture.png" width="800" title="NN_architecture">