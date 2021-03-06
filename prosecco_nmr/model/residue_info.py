import numpy as np
from ..database import atom_names

__all__ = ['RESIDUES','BLOSUM62',"BACKBONE_ATOMS","ALL_ATOMS"]

BACKBONE_ATOMS = [ "CA", "CB", "C", "H", "HA", "N"]

ALL_ATOMS = sorted(atom_names)

RESIDUES = ["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]

BLOSUM62 = np.array([[ 4.,  0., -2., -1., -2.,  0., -2., -1., -1., -1., -1., -2., -1.,
        -1., -1.,  1.,  0.,  0., -3., -2.],
       [ 0.,  9., -3., -4., -2., -3., -3., -1., -3., -1., -1., -3., -3.,
        -3., -3., -1., -1., -1., -2., -2.],
       [-2., -3.,  6.,  2., -3., -1., -1., -3., -1., -4., -3.,  1., -1.,
         0., -2.,  0., -1., -3., -4., -3.],
       [-1., -4.,  2.,  5., -3., -2.,  0., -3.,  1., -3., -2.,  0., -1.,
         2.,  0.,  0., -1., -2., -3., -2.],
       [-2., -2., -3., -3.,  6., -3., -1.,  0., -3.,  0.,  0., -3., -4.,
        -3., -3., -2., -2., -1.,  1.,  3.],
       [ 0., -3., -1., -2., -3.,  6., -2., -4., -2., -4., -3.,  0., -2.,
        -2., -2.,  0., -2., -3., -2., -3.],
       [-2., -3., -1.,  0., -1., -2.,  8., -3., -1., -3., -2.,  1., -2.,
         0.,  0., -1., -2., -3., -2.,  2.],
       [-1., -1., -3., -3.,  0., -4., -3.,  4., -3.,  2.,  1., -3., -3.,
        -3., -3., -2., -1.,  3., -3., -1.],
       [-1., -3., -1.,  1., -3., -2., -1., -3.,  5., -2., -1.,  0., -1.,
         1.,  2.,  0., -1., -2., -3., -2.],
       [-1., -1., -4., -3.,  0., -4., -3.,  2., -2.,  4.,  2., -3., -3.,
        -2., -2., -2., -1.,  1., -2., -1.],
       [-1., -1., -3., -2.,  0., -3., -2.,  1., -1.,  2.,  5., -2., -2.,
         0., -1., -1., -1.,  1., -1., -1.],
       [-2., -3.,  1.,  0., -3.,  0.,  1., -3.,  0., -3., -2.,  6., -2.,
         0.,  0.,  1.,  0., -3., -4., -2.],
       [-1., -3., -1., -1., -4., -2., -2., -3., -1., -3., -2., -2.,  7.,
        -1., -2., -1., -1., -2., -4., -3.],
       [-1., -3.,  0.,  2., -3., -2.,  0., -3.,  1., -2.,  0.,  0., -1.,
         5.,  1.,  0., -1., -2., -2., -1.],
       [-1., -3., -2.,  0., -3., -2.,  0., -3.,  2., -2., -1.,  0., -2.,
         1.,  5., -1., -1., -3., -3., -2.],
       [ 1., -1.,  0.,  0., -2.,  0., -1., -2.,  0., -2., -1.,  1., -1.,
         0., -1.,  4.,  1., -2., -3., -2.],
       [ 0., -1., -1., -1., -2., -2., -2., -1., -1., -1., -1.,  0., -1.,
        -1., -1.,  1.,  5.,  0., -2., -2.],
       [ 0., -1., -3., -2., -1., -3., -3.,  3., -2.,  1.,  1., -3., -2.,
        -2., -3., -2.,  0.,  4., -3., -1.],
       [-3., -2., -4., -3.,  1., -2., -2., -3., -3., -2., -1., -4., -4.,
        -2., -3., -3., -2., -3., 11.,  2.],
       [-2., -2., -3., -2.,  3., -3.,  2., -1., -2., -1., -1., -2., -3.,
        -1., -2., -2., -2., -1.,  2.,  7.]])
