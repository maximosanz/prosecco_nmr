import numpy as np
import subprocess

from ..model import make_test_arr

from pathlib import Path

__all__ = ['input_fromseq',
	'parse_fasta',
	'make_NN_arrays',
	'make_PROSECCO_nn',
	'make_test_arr',
	'Residue_Scaler',
	'load_scaler'
	]


def run_spartaplus(pdbid,
	SPARTA_directory="BENCHMARK/SPARTA+",
	PDB_directory="./PDB",
	PDB_prefix="",
	PDB_suffix='.pdb',
	spartaplus_prefix="",
	spartaplus_suffix='.tab',
	spartaplus_exe="sparta+"):
	
	ind = Path(PDB_directory)
	outd = Path(SPARTA_directory)
	outd.mkdir(parents=True, exist_ok=True)

	infn = Path(d / (PDB_prefix+pdbid+PDB_suffix))
	outfn = Path(d / (spartaplus_prefix+pdbid+spartaplus_suffix))

	subprocess.run([spartaplus_exe,
		"-in", str(infn),
		"-out",str(outfn)
		])

	return
