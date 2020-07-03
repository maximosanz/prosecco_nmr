import numpy as np
import subprocess

from ..model import make_test_arr

from pathlib import Path

__all__ = ['run_spartaplus'
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

	infn = Path(ind / (PDB_prefix+pdbid+PDB_suffix))
	outfn = Path(outd / (spartaplus_prefix+pdbid+spartaplus_suffix))

	if not infn.is_file():
		warnings.warn("Missing file: {}".format(str(infn)))
		return 

	subprocess.run([spartaplus_exe,
		"-in", str(infn),
		"-out",str(outfn)
		])

	return
