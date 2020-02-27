import numpy as np
import pandas as pd
import warnings

__all__ = ['make_NN_arrays'
	]

def make_NN_arrays(EntryDB,
	CSdb,
	BLOSUM62=True,
	atoms=None,
	seq_neigh=2,
	SS_neigh=3,
	SS_type="PSIPRED"):

	N = len(EntryDB)
	for i,entry in EntryDB.iterrows():
		eID = entry["BMRB_ID"]
		perc = int(i*100/N)
		print("\r  >> Making NN input: Entry {} ; Progress: {}/{} ({}%)     ".format(eID,i,N,perc), end='')
		seq = entry["Sequence"]
		eCS = CSdb[CSdb["BMRB_ID"]==eID]
		for j,cs in eCS.iterrows():
			pos = cs["Res_ID"]-1
			res = cs["Residue"]
			# Check again if it matches - although this has been done at the database building stage
			if res != seq[pos]:
				warnings.warn("Sequence mismatch for entry {} at position {} ({} - {})".format(str(eID),pos+1,res,seq[pos]))
				continue
			if res == "X":
				continue

