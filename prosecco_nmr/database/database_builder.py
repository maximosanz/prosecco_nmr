import requests
import pynmrstar
import warnings
import numpy as np
import pandas as pd
import re
import subprocess
import pkg_resources
import os
import sys

import Bio.PDB

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import scale
from sklearn.cluster import AgglomerativeClustering
from Bio.SeqUtils import IUPACData
from pathlib import Path

__all__ = ['get_BMRB_entries',
	'get_NMRSTAR_files',
	'get_PDB_files',
	'build_entry_database',
	'remove_entries',
	'cluster_sequences',
	'build_CS_database',
	'remove_outliers',
	'cluster_cystines',
	'cluster_transP'
	]

__MY_APPLICATION__ = "PROSECCO-NMR"

def _extract_experimental_conditions(nmrstar,experimental_conditions=["ph","temperature"]):

	conditions = nmrstar.get_tag("_Sample_condition_variable.Type")
	conditions_val = nmrstar.get_tag("_Sample_condition_variable.Val")
	# Make it lowercase to allow different capitalisations
	conditions = [s.lower() for s in conditions]
	conditions_d = {c : np.nan for c in experimental_conditions}

	sample_type = nmrstar.get_tag("_Sample.Type")[0].lower()
	# Micelles is found spelled in singular or plural - keep only singular
	if sample_type == 'micelles':
		sample_type = sample_type[:-1]

	conditions_d["sample_type"] = sample_type
	
	for cond in experimental_conditions:
		if cond not in conditions:
			continue
		val = conditions_val[conditions.index(cond)]
		try:
			val = float(val)
			conditions_d[cond] = val
		except ValueError:
			continue

	return conditions_d

def _extract_sequences(nmrstar,entities):
	seqd = nmrstar.get_tags(["_Entity.ID",
		"_Entity.Name",
		"_Entity.Polymer_seq_one_letter_code"])
	seqs = {}
	for entity in entities:
		entityIDX = seqd["_Entity.ID"].index(entity)
		seq = seqd["_Entity.Polymer_seq_one_letter_code"][entityIDX]
		seq = seq.replace("\n","")
		seqs[entity] = seq
	return seqs

def _is_denatured(nmrstar):
	denatured = False
	buffercomp = nmrstar.get_tag("_Sample_component.Mol_common_name")
	for compi,comp in enumerate(buffercomp):
		if re.search("urea|guanidine",comp.lower()):
			denatured = True
			break
	return denatured

def _check_local_dir(d):
	d_path = Path(d)
	if not d_path.is_dir():
		raise ValueError('Missing NMRSTAR directory {}'.format(d))
	return

def _check_local_filepath(d,fn):
	d_path = Path(d)
	f_path = Path(d_path / fn )
	if not f_path.is_file():
		warnings.warn("Missing file: {}".format(str(f_path)))
		return None
	return str(f_path)

def _get_NMRSTAR(eID,
	local_files=True,
	NMRSTAR_prefix="",
	NMRSTAR_suffix=".str",
	NMRSTAR_directory="./NMRSTAR"):
	if local_files:
		fn = NMRSTAR_prefix+str(eID)+NMRSTAR_suffix
		f = _check_local_filepath(NMRSTAR_directory,fn)
		if f is None:
			return None
		nmrstar = pynmrstar.Entry.from_file(str(f))
	else:
		nmrstar = pynmrstar.Entry.from_database(eID)
	return nmrstar

def _parse_nomenclature_table(fn="atom_nom.tbl"):
	#path = 'classifiers/text_cat/README.md'  # always use slash
	filepath = pkg_resources.resource_filename(__name__, fn)
	d = {}
	for l in open(filepath):
		# This ignores terminal protons
		if not l.strip() or l[0] in ["#","X"]:
			continue
		c = l.split('\t')
		res = c[0]
		if res not in d:
			d[res] = {}
		bmrb = c[2]
		pdb = c[4]
		d[res][bmrb] = pdb
	return d


def get_BMRB_entries():
	base_link = "http://webapi.bmrb.wisc.edu/v2/"
	macro_entries = requests.get(base_link+"/list_entries?database=macromolecules",
		headers={"Application":__MY_APPLICATION__})
	entries = macro_entries.json()
	return entries

def get_NMRSTAR_files(entries,directory="./NMRSTAR",prefix="",suffix=".str"):
	d = Path(directory)
	d.mkdir(parents=True, exist_ok=True)
	N = len(entries)
	for i, eID in enumerate(entries):
		fn = Path(d / (prefix+eID+suffix))
		# Ignore already downloaded files
		if fn.is_file():
			continue
		perc = int(i*100/N)
		print("\r  >> Downloading NMRSTAR file for BMRB Entry {} Progress: {}/{} ({}%)     ".format(eID,i,N,perc), end='')
		entry = pynmrstar.Entry.from_database(eID)
		entry.write_to_file(str(fn))
	return

def get_PDB_files(PDBs,directory="./PDB",prefix="",suffix='.pdb'):
	d = Path(directory)
	d.mkdir(parents=True, exist_ok=True)
	N = len(PDBs)
	pdbl = Bio.PDB.PDBList()
	for i, pdb_ID in enumerate(PDBs):
		pdb_fn = Path(d / (prefix+pdb_ID+suffix))
		# Ignore already downloaded files
		if pdb_fn.is_file():
			continue
		perc = int(i*100/N)
		print("\r  >> Downloading PDB file {} Progress: {}/{} ({}%)     ".format(pdb_ID,i,N,perc), end='')
		old_stdout = sys.stdout
		sys.stdout = open(os.devnull, "w")
		pdbl.download_pdb_files([pdb_ID], pdir=str(directory), file_format='pdb',obsolete=False)
		sys.stdout = old_stdout
		# Biopython uses a strange file naming...
		downloaded_fn = Path(d / ("pdb"+pdb_ID.lower()+".ent"))
		if downloaded_fn.is_file():
			os.rename(downloaded_fn, pdb_fn)
	return

def build_entry_database(entries,
	local_files=True,
	NMRSTAR_directory="./NMRSTAR",
	NMRSTAR_prefix="",
	NMRSTAR_suffix=".str",
	PDBmatch_file=None,
	experimental_conditions=["ph","temperature","pressure"],
	record_denaturant=True):

	'''
	Build a pandas dataframe containing the information on the BMRB entries listed in "entries"
	
	This is by default built from local files (much faster than looking them up online - which can also be done)
	'''

	if PDBmatch_file is None:
		PDB_match = requests.get("http://webapi.bmrb.wisc.edu/v2/mappings/bmrb/pdb?format=text&match_type=exact",
			headers={"Application":__MY_APPLICATION__}).text
	else:
		PDB_match = open(PDBmatch_file).read()
	PDBmatch_d = { l.split()[0] : [ x for x in l.split()[1].split(',') ] for l in PDB_match.split('\n')}
	if local_files:
		_check_local_dir(NMRSTAR_directory)

	EntryDB = []
	N = len(entries)
	for i, eID in enumerate(entries):
		perc = int(i*100/N)
		print("\r  >> Building database: Entry {} ; Progress: {}/{} ({}%)     ".format(eID,i,N,perc), end='')

		nmrstar = _get_NMRSTAR(eID,
			local_files=local_files,
			NMRSTAR_prefix=NMRSTAR_prefix,
			NMRSTAR_suffix=NMRSTAR_suffix,
			NMRSTAR_directory=NMRSTAR_directory)
		if nmrstar is None:
			continue

		CS = nmrstar.get_loops_by_category("Atom_chem_shift")

		# Ignore entries with no chemical shift information:
		if not len(CS):
			continue

		entities = set(CS[0].get_tag("Entity_ID"))
		# Ignore entries with more than one "entity"
		# This could be rectified by more careful parsing
		if len(entities) > 1:
			continue

		# Ignore nucleotide entries
		polymer_type = nmrstar.get_tag("Entity.Polymer_type")[0].lower()
		if re.search("nucleotide",polymer_type):
			continue

		nCS = len(CS[0])
		# Parse experimental conditions:
		# Sample type and other conditions listed in experimental_conditions
		conditions_d = _extract_experimental_conditions(nmrstar,experimental_conditions=experimental_conditions)
		seqs = _extract_sequences(nmrstar,entities)

		# Since we have only one entity - we have a single sequence:
		seq = list(seqs.values())[0].upper()

		# Taking only the first PDB match
		pdb = "XXXX"
		if eID in PDBmatch_d:
			pdb = PDBmatch_d[eID][0]

		Entry_d = {"BMRB_ID" : eID,
			"N_CS" : nCS,
			"Sequence" : seq,
			"PDB_ID" : pdb,
			"polymer_type" : polymer_type
			}

		for cond in conditions_d:
			Entry_d[cond] = conditions_d[cond]
		if record_denaturant:
			Entry_d["Denatured"] = _is_denatured(nmrstar)
		EntryDB.append(Entry_d)

	EntryDB = pd.DataFrame(EntryDB)
	return EntryDB


def remove_entries(EntryDB,
	Trange=(273.,333.),
	pHrange=(3.,11.),
	Prange=(0.,1.5),
	remove_denatured=True,
	keep_NaNs=True,
	reset_index=True):
	# Entries should be kept by default if missing data
	# Especially important for pressure, which is often missing

	conditions = ["temperature","ph","pressure"]
	ranges = [Trange,pHrange,Prange]
	for i,cond in enumerate(conditions):
		lims = ranges[i]
		idx = (EntryDB[cond] > lims[0]) & (EntryDB[cond] < lims[1])
		if keep_NaNs:
			idx = idx | np.isnan(EntryDB[cond])
		EntryDB = EntryDB[idx]
	if remove_denatured:
		EntryDB = EntryDB[EntryDB["Denatured"]==False]
	if reset_index:
		EntryDB = EntryDB.reset_index(drop=True)
	return EntryDB

def cluster_sequences(EntryDB,
	similarity=0.9,
	usearch_exe="usearch",
	seq_fn="entries.fasta",
	centroids_fn="centroids.fasta",
	clusters_fn="clusters.uc",
	reset_index=True):

	'''
	Cluster the sequences using the UCLUST algorithm
	and return a database with only the cluster centroids

	The usearch executable must be accessible, it can be downloaded from:
	https://drive5.com/usearch/ - 2020-02-11

	UCLUST is a greedy algorithm that tries to find cluster centroid first - so the order of sequences matters
	I will list sequences with a PDB match first, in descending order of number of chemical shift data
	'''

	seqs = []
	IDs = []

	idx = EntryDB["PDB_ID"] == "XXXX"
	for i in range(2):
		idx = ~idx
		subset = EntryDB[idx].sort_values("N_CS",ascending=False)
		seqs.extend(list(subset["Sequence"].values))
		IDs.extend(list(subset["BMRB_ID"].values))

	seqf = open(seq_fn,'w')

	for i,seq in enumerate(seqs):
		seqf.write(">{}\n".format(IDs[i]))
		n=60
		splitseq = [ seq[i:i+n] for i in range(0,len(seq),n) ]
		[ seqf.write(segment+'\n') for segment in splitseq ]

	seqf.close()

	subprocess.run([usearch_exe,
		"-cluster_fast", seq_fn,
		"-id", str(similarity),
		"-centroids",centroids_fn,
		"-uc",clusters_fn
		])

	centroids = [ l[1:].strip() for l in open(centroids_fn) if l[0] == ">" ]
	EntryDB = EntryDB[np.isin(EntryDB["BMRB_ID"],centroids)]

	if reset_index:
		EntryDB = EntryDB.reset_index(drop=True)

	return EntryDB


def build_CS_database(EntryDB,
	local_files=True,
	NMRSTAR_directory="./NMRSTAR",
	NMRSTAR_prefix="",
	NMRSTAR_suffix=".str",
	tolerance=0.01,
	return_discarded=False):

	if local_files:
		_check_local_dir(NMRSTAR_directory)

	atom_nom = _parse_nomenclature_table()

	CS_db = []
	N = len(EntryDB)
	discarded = []

	for i, entry in EntryDB.iterrows():
		eID = entry["BMRB_ID"]
		perc = int(i*100/N)
		print("\r  >> Building CS database: Entry {} ; Progress: {}/{} ({}%)     ".format(eID,i,N,perc), end='')
		nmrstar = _get_NMRSTAR(eID,
			local_files=local_files,
			NMRSTAR_prefix=NMRSTAR_prefix,
			NMRSTAR_suffix=NMRSTAR_suffix,
			NMRSTAR_directory=NMRSTAR_directory)
		if nmrstar is None:
			continue

		cs_result_sets = []
		for chemical_shift_loop in nmrstar.get_loops_by_category("Atom_chem_shift"):
			cs_result_sets.append(chemical_shift_loop.get_tag(['Entity_ID',
				'Comp_index_ID',
				'Comp_ID',
				'Atom_ID',
				'Atom_type',
				'Val',
				'Val_err']))

		# Ignore entries with more than 1 CS set - there shouldn't be any as they were discarded before
		if len(cs_result_sets) > 1:
			continue

		seq = entry["Sequence"]
		Entry_CS = [ {"BMRB_ID":eID, 
			"Res_ID": i+1,
			"Residue":seq[i]}
			for i in range(len(seq)) ]

		CS = cs_result_sets[0]
		discard = False
		for x in CS:
			CS_resid = int(x[1])
			pos = CS_resid-1
			res = x[2]
			res = res[0]+res[1:].lower()
			try:
				res_1let = IUPACData.protein_letters_3to1[res].upper()
			except KeyError:
				res_1let = "X"
			res_to_match = seq[pos]
			if res_1let != res_to_match:
				# Discard entry if sequence mismatch
				discarded.append(eID)
				discard = True
				break
			# Skip residues labeled X
			if res_1let == "X":
				continue
			at = x[3]
			if res_1let not in atom_nom or at not in atom_nom[res_1let]:
				# Atoms not in the standard nomenclature
				continue
			try:
				cs_val = float(x[5])
			except ValueError:
				continue
			try:
				cs_err = float(x[6])
			except ValueError:
				cs_err = np.nan
			if at in Entry_CS[pos]:
				if abs(cs_val - Entry_CS[pos][at]) > tolerance:
					# Discard entry if multiple CS for the same atom
					discarded.append(eID)
					discard = True
					break
			Entry_CS[pos][at] = cs_val
			Entry_CS[pos][at+"_Err"] = cs_err

		if discard:
			continue
		CS_db.extend(Entry_CS)

	CS_db = pd.DataFrame(CS_db)

	if return_discarded:
		return CS_db,discarded
	return CS_db

def remove_outliers(CS_db,contamination=0.01):
	atom_nom = _parse_nomenclature_table()
	for i,res in enumerate(atom_nom):
		print("\r  >> Removing outliers for residue {} ({} / {})     ".format(res,i+1,len(atom_nom)), end='')
		res_IDX = CS_db['Residue'] == res
		for at in atom_nom[res]:
			x = CS_db[res_IDX]
			if at not in x.columns:
				continue
			OutlierModel = IsolationForest(contamination=contamination,bootstrap=True)
			at_cs = x[at].values
			noNaN_IDX = ~np.isnan(at_cs)
			at_cs_noNaN = at_cs[noNaN_IDX]
			at_cs_pred = at_cs_noNaN.reshape(-1,1)
			outlier_pred = OutlierModel.fit_predict(at_cs_pred)
			outliers = (outlier_pred-1).astype(bool)
			at_cs_noNaN[outliers] = np.nan
			at_cs[noNaN_IDX] = at_cs_noNaN
			CS_db.loc[res_IDX,at] = at_cs
	return CS_db

def cluster_cystines(CS_db,atoms=["CA","CB"]):
	return _cluster_byCS(CS_db,"C",atoms,"Cystine")

def cluster_transP(CS_db,atoms=["CB","CG","CD"]):
	return _cluster_byCS(CS_db,"P",atoms,"Trans",n_clusters=3,true_idx=2)

def _cluster_byCS(CS_db,residue,atoms,attribute,n_clusters=2,true_idx=1,linkage='ward'):
	CS_db[attribute] = False
	Res_Idx = CS_db["Residue"] == residue
	Res_CS = CS_db[Res_Idx]
	Res_CS_X_all = np.array([ Res_CS[at] for at in atoms ]).T
	NaN_IDX = np.isnan(Res_CS_X_all).any(axis=1)
	Res_CS_X = Res_CS_X_all[~NaN_IDX,:]
	Res_CS_X = scale(Res_CS_X)
	agg = AgglomerativeClustering(n_clusters=n_clusters,linkage=linkage).fit(Res_CS_X)
	labels = agg.labels_
	unique_labels, counts = np.unique(labels,return_counts=True)
	true_attr = np.zeros(Res_CS_X_all.shape[0],dtype=bool)
	true_attr[~NaN_IDX] = (labels == true_idx)
	CS_db.loc[Res_Idx,attribute] = true_attr
	return CS_db

