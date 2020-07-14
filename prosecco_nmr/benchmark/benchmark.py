import numpy as np
import subprocess
import re
import warnings

from ..model import make_test_arr, BACKBONE_ATOMS

from pathlib import Path

__all__ = ['run_spartaplus',
	'make_experimental_arr',
	'make_predictor_arr',
	'SPARTAplus_parser',
	'compute_RMSDs'
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

def make_experimental_arr(Entries,CS_db,entry_list=None,year=None,atoms=BACKBONE_ATOMS,exclude_noPDB=False):
	# Returns experimental CS array [N_entries,max_seq_len,N_Atoms], BMRB_map and PDB_map
	# entry_list takes priority over year
	N_Atoms = len(atoms)
	if entry_list is not None:
		becnhmark_entries = Entries[np.isin(Entries["BMRB_ID"],entry_list)]
	elif year is not None:
		becnhmark_entries = Entries[Entries["Year"] == year]
	else:
		becnhmark_entries = Entries.copy()
	if exclude_noPDB:
		becnhmark_entries = becnhmark_entries[becnhmark_entries["PDB_ID"] != "XXXX"]
	entry_list = list(becnhmark_entries["BMRB_ID"])
	becnhmark_CS = CS_db[np.isin(CS_db["BMRB_ID"],entry_list)]

	becnhmark_entries.reset_index(inplace=True)
	becnhmark_CS.reset_index(inplace=True)

	N_Entries = len(entry_list)
	max_seq_len = max([ len(seq) for seq in becnhmark_entries["Sequence"] ])
	Exp_Arr = np.zeros((N_Entries,max_seq_len,N_Atoms))
	Exp_Arr[:] = np.nan
	PDB_map = list(becnhmark_entries["PDB_ID"])

	for i, CS_row in becnhmark_CS.iterrows():
		eIDX = entry_list.index(CS_row["BMRB_ID"])
		Exp_Arr[eIDX,CS_row["Res_ID"]-1] = CS_row[atoms]

	return Exp_Arr, entry_list, PDB_map

def SPARTAplus_parser(pdbid,
	atoms=BACKBONE_ATOMS,
	SPARTA_directory="BENCHMARK/SPARTA+",
	spartaplus_prefix="",
	spartaplus_suffix='.tab',
	change_proton_nomenclature=True):

	d = Path(SPARTA_directory)
	fn = Path(d / (spartaplus_prefix+pdbid+spartaplus_suffix))
	if not fn.is_file():
		warnings.warn("Missing file: {}".format(str(fn)))
		return 
	f = open(fn)

	cs_arr = []
	curres = 0
	for l in f:
		l = l.strip()
		if not l:
			continue
		if re.match("REMARK|DATA|VARS|FORMAT",l):
			continue
		c = l.split()
		resid = int(c[0])
		if resid != curres:
			empty = np.zeros(len(atoms))
			empty[:] = np.nan
			cs_arr.append(empty)
			curres = resid
		res = c[1]
		atom = c[2]
		cs = float(c[4])
		# change sparta amide proton nomenclature
		if change_proton_nomenclature and atom == "HN":
			atom = "H"
		if atom not in atoms:
			continue
		atomidx = atoms.index(atom)
		cs_arr[-1][atomidx] = cs
	return np.array(cs_arr)

_PARSER_d = {
	"SPARTA+" : SPARTAplus_parser
}

def make_predictor_arr(Entries,predictor="SPARTA+",entry_list=None,atoms=BACKBONE_ATOMS):
	parser = _PARSER_d[predictor]
	N_Atoms = len(atoms)
	if entry_list is not None:
		becnhmark_entries = Entries[np.isin(Entries["BMRB_ID"],entry_list)]
	else:
		becnhmark_entries = Entries.copy()
		entry_list = list(becnhmark_entries["BMRB_ID"])
	N_Entries = len(entry_list)
	max_seq_len = max([ len(seq) for seq in becnhmark_entries["Sequence"] ])
	Pred_Arr = np.zeros((N_Entries,max_seq_len,N_Atoms))
	Pred_Arr[:] = np.nan

	for i, eID in enumerate(entry_list):
		Entry = Entries[Entries["BMRB_ID"] == eID].iloc[0]
		pdbid = Entry["PDB_ID"]
		PDB_match_start = Entry["PDB_Match_Start"]
		BMRB_match_start = Entry["BMRB_Match_Start"]
		PDB_match_len = Entry["PDB_Match_Length"]
		eCS = parser(pdbid)
		if eCS is None or eCS.shape[0] < PDB_match_len:
			continue
		Pred_Arr[i,BMRB_match_start:BMRB_match_start+PDB_match_len] = eCS[PDB_match_start:PDB_match_start+PDB_match_len]

	return Pred_Arr


def compute_RMSDs(array_list,experimental_idx=0):
	# Move experimental set to first position before stacking
	array_list = [array_list[experimental_idx]] + array_list[:experimental_idx] + array_list[experimental_idx+1:]
	stack = np.stack(array_list)
	stack = stack.reshape((stack.shape[0],-1,stack.shape[-1]))
	# Remove CS that are NaN for ANY predictor so that the comparison is fair (exact same set of CS for all)
	anyNaN = np.any(np.isnan(stack),0)
	stack[:,anyNaN] = np.nan
	rmsd = np.sqrt( np.nanmean( (stack[0] - stack[1:])**2, 1) )
	return rmsd
















