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
	'get_all_PDB_files',
	'build_entry_database',
	'remove_entries',
	'cluster_sequences',
	'build_CS_database',
	'include_PSIPRED',
	'PSIPRED_seq',
	'run_PSIPRED',
	'parse_ss2',
	'remove_outliers',
	'cluster_cystines',
	'cluster_transPRO',
	'cluster_protHIS',
	'check_referencing',
	'keep_entries',
	'atom_names',
	'pick_cluster_centers',
	'RESIDUE_SIDECHAIN_GROUPS',
	'RESIDUE_IGNORE_ATOMS'
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
		raise ValueError('Missing directory {}'.format(d))
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

def _get_residue_atoms(fn="atom_nom.tbl"):
	d = _parse_nomenclature_table(fn)
	d2 = { k : [ k2 for k2, v2 in v.items() ] for k,v in d.items() }
	return d2

RESIDUE_SIDECHAIN_GROUPS = {
	"A" : {"HB" : ["HB1", "HB2", "HB3"]},
	"C" : {"HB-23" : ["HB2", "HB3"]},
	"D" : {"HB-23" : ["HB2", "HB3"]},
	"E" : {"HB-23" : ["HB2", "HB3"],
		"HG-23" : ["HG2", "HG3"]},
	"F" : {"HB-23" : ["HB2", "HB3"],
		"CD-12" : ["CD1", "CD2"],
		"HD-12" : ["HD1", "HD2"],
		"CE-12" : ["CE1", "CE2"],
		"HE-12" : ["HE1", "HE2"]},
	"G" : {"HA-23" : ["HA2", "HA3"]},
	"H" : {"HB-23" : ["HB2", "HB3"]},
	"I" : {"HG1-23" : ["HG12", "HG13"],
		"HG2" : ["HG21", "HG22", "HG23"],
		"HD1" : ["HD11", "HD12", "HD13"]},
	"K" : {"HB-23" : ["HB2", "HB3"],
		"HG-23" : ["HG2", "HG3"],
		"HD-23" : ["HD2", "HD3"],
		"HE-23" : ["HE2", "HE3"]},
	"L" : {"HB-23" : ["HB2", "HB3"],
		"HD1" : ["HD11", "HD12", "HD13"],
		"HD2" : ["HD21", "HD22", "HD23"]},
	"M" : {"HB-23" : ["HB2", "HB3"],
		"HG-23" : ["HG2", "HG3"],
		"HE" : ["HE1", "HE2", "HE3"]},
	"N" : {"HB-23" : ["HB2", "HB3"],
		"HD2-12" : ["HD21", "HD22"]},
	"P" : {"HB-23" : ["HB2", "HB3"],
		"HG-23" : ["HG2", "HG3"],
		"HD-23" : ["HD2", "HD3"]},
	"Q" : {"HB-23" : ["HB2", "HB3"],
		"HG-23" : ["HG2", "HG3"],
		"HE2-12" : ["HE21", "HE22"]},
	"R" : {"HB-23" : ["HB2", "HB3"],
		"HG-23" : ["HG2", "HG3"],
		"HD-23" : ["HD2", "HD3"]},
	"S" : {"HB-23" : ["HB2", "HB3"]},
	"T" : {"HG2" : ["HG21", "HG22", "HG23"]},
	"V" : {"HG1" : ["HG11", "HG12", "HG13"],
		"HG2" : ["HG21", "HG22", "HG23"]},
	"W" : {"HB-23" : ["HB2", "HB3"]},
	"Y" : {"HB-23" : ["HB2", "HB3"],
		"CD-12" : ["CD1", "CD2"],
		"HD-12" : ["HD1", "HD2"],
		"CE-12" : ["CE1", "CE2"],
		"HE-12" : ["HE1", "HE2"]}
}

RESIDUE_IGNORE_ATOMS = {
	"A" : ["O"],
	"C" : ["O", "SG"],
	"D" : ["O", "CG", "OD1", "OD2", "HD2"],
	"E" : ["O", "CD", "OE1", "OE2", "HE2"],
	"F" : ["O", "CG"],
	"G" : ["O"],
	"H" : ["O", "HD1", "HE2", "CG", "ND1", "NE2"],
	"I" : ["O"],
	"K" : ["O", "HZ1", "HZ2", "HZ3", "NZ"],
	"L" : ["O"],
	"M" : ["O"],
	"N" : ["O", "OD1"],
	"P" : ["O", "H2", "H3"],
	"Q" : ["O", "OE1"],
	"R" : ["O", "HH11", "HH12", "HH21", "HH22", "CZ", "NE", "NH1", "NH2"],
	"S" : ["O", "HG", "OG"],
	"T" : ["O", "OG1"],
	"V" : ["O"],
	"W" : ["O", "CG", "CD2", "CE2"],
	"Y" : ["O", "HH", "CG", "CZ", "OH"]
}

def _get_all_atnames(fn="atom_nom.tbl"):
	d = _parse_nomenclature_table(fn)
	ats = set([])
	for key, value in d.items():
		ats.update(value)
	return list(ats)

atom_names = _get_all_atnames()

def _get_PROSECCO_atnames():
	d = _parse_nomenclature_table()
	PROSECCO_d = {}
	for res,atomd in d.items():
		grouped_ats = []
		grouped_atnames = []
		for k,v in RESIDUE_SIDECHAIN_GROUPS[res].items():
			grouped_ats.extend(v)
			grouped_ats.append(k)
		PROSECCO_d[res] = []
		for atom in atomd:
			if atom in RESIDUE_IGNORE_ATOMS[res] or atom in grouped_ats:
				continue
			PROSECCO_d[res].append(atom)
	return PROSECCO_d

def get_atname_list():
	atname_d = _get_PROSECCO_atnames()
	list(set([ at for res, ats in atname_d for at in ats ]))


def _dump_seq(f,seq,seqID,n=60):
	f.write(">{}\n".format(seqID))
	splitseq = [ seq[i:i+n] for i in range(0,len(seq),n) ]
	[ f.write(segment+'\n') for segment in splitseq ]
	return

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

def get_all_PDB_files(directory="./PDB",prefix="",suffix='.pdb'):
	PDB_match = requests.get("http://webapi.bmrb.wisc.edu/v2/mappings/bmrb/pdb?format=text&match_type=exact",
			headers={"Application":__MY_APPLICATION__}).text
	PDBmatch_d = { l.split()[0] : [ x for x in l.split()[1].split(',') ] for l in PDB_match.split('\n') }
	PDBs = []
	for BMRB, pdb in PDBmatch_d.items():
		PDBs.extend(pdb)
	get_PDB_files(PDBs,directory=directory,prefix=prefix,suffix=suffix)
	return

def build_entry_database(entries,
	local_files=True,
	NMRSTAR_directory="./NMRSTAR",
	NMRSTAR_prefix="",
	NMRSTAR_suffix=".str",
	PDBmatch_file=None,
	experimental_conditions=["ph","temperature","pressure"],
	record_denaturant=True,
	local_PDB_files=True,
	PDB_directory="./PDB",
	PDB_prefix="",
	PDB_suffix=".pdb",
	remove_if_X=0.8):

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
	if local_PDB_files:
		_check_local_dir(PDB_directory)
		PDB_parser = Bio.PDB.PDBParser(QUIET=True)

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
		if re.search("nucleotide|polysaccharide|other|dna|cyclic",polymer_type):
			continue
		
		try:
			release = nmrstar.get_tag("Release.Date")
			years = [ int(d.split("-")[0]) for d in release ]
			release_year = min(years)
		except KeyError:
			release_year = np.nan

		nCS = len(CS[0])
		# Parse experimental conditions:
		# Sample type and other conditions listed in experimental_conditions
		conditions_d = _extract_experimental_conditions(nmrstar,experimental_conditions=experimental_conditions)
		seqs = _extract_sequences(nmrstar,entities)

		# Since we have only one entity - we have a single sequence:
		seq = list(seqs.values())[0].upper()

		# We will ignore the entry if the fraction of "X" residues is more than remove_if_X
		N_X = len([ "X" for ch in seq if ch == "X"])
		f_X = float(N_X)/len(seq)
		if f_X > remove_if_X:
			continue

		# Taking the first matching PDB
		pdb = "XXXX"
		PDB_match = (0,0,0,0)
		dssp = np.nan
		eID_str = str(eID)
		if eID_str in PDBmatch_d:
			if not local_PDB_files:
				pdb = PDBmatch_d[eID_str][0]
			else:
				for pdb in PDBmatch_d[eID_str]:
					PDB_match = None
					fn = Path(PDB_directory) / Path(PDB_prefix+pdb.upper()+PDB_suffix)
					if fn.is_file():
						structure = PDB_parser.get_structure(pdb, str(fn))
						PDB_match = _match_PDBseq(structure,seq)
						if PDB_match is not None:
							dssp = _get_DSSP(structure,str(fn),PDB_match[3])
							if dssp is None:
								PDB_match = None
								continue
							dssp = "X"*PDB_match[1]+dssp[PDB_match[0]:PDB_match[0]+PDB_match[2]]
							dssp += "X"*(len(seq)-len(dssp))
							break
			if PDB_match is None:
				pdb = "XXXX"
				PDB_match = (0,0,0,0)

		Entry_d = {"BMRB_ID" : eID,
			"Year" : release_year,
			"N_CS" : nCS,
			"Sequence" : seq,
			"PDB_ID" : pdb,
			"polymer_type" : polymer_type,
			"PDB_Chain_Match" : PDB_match[3],
			"PDB_Match_Start" : PDB_match[0],
			"BMRB_Match_Start" : PDB_match[1],
			"PDB_Match_Length" : PDB_match[2],
			"DSSP" : dssp
			}

		for cond in conditions_d:
			Entry_d[cond] = conditions_d[cond]
		if record_denaturant:
			Entry_d["Denatured"] = _is_denatured(nmrstar)
		EntryDB.append(Entry_d)

	EntryDB = pd.DataFrame(EntryDB)
	return EntryDB

def _get_DSSP(PDB_structure,fn,chain):
	model = PDB_structure[0]
	try:
		dssp = Bio.PDB.DSSP(model,fn)
		dssp_str = "".join([ dssp[k][2] for k in dssp.keys() if k[0] == chain ] )
	except:
		dssp_str = None
	return dssp_str

def _match_PDBseq(PDB_structure,seq):
	model = PDB_structure[0]
	# First_match is a len=4 tuple that contains the first position matching in the PDB, in the BMRB,
	# the length of the match and the PDB chain ID
	First_match = None
	for chain in model:
		PDB_seq = ''
		for residue in chain:
			r = residue.get_resname()
			r = r[0].upper()+r[1:].lower()
			try:
				r = IUPACData.protein_letters_3to1[r].upper()
			except KeyError:
				r = "X"
			PDB_seq += r
			# Removing C-terminal "X" residues if any
			PDB_seq = PDB_seq.rstrip("X")
		if re.search(PDB_seq,seq):
			First_match = (0,re.search(PDB_seq,seq).start(),len(PDB_seq),chain.id)
			return First_match
		elif re.search(seq,PDB_seq):
			First_match = (re.search(seq,PDB_seq).start(),0,len(seq),chain.id)
			return First_match
	return None



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
	distmx_fn="distmx.txt",
	clusters_fn="clusters.uc",
	linkage="min"):
	'''
	Cluster the sequences using the UCLUST algorithm
	and return a database with cluster labels
	Agglomerative clustering with single linkage

	The usearch executable must be accessible, it can be downloaded from:
	https://drive5.com/usearch/ - 2020-02-11
	'''
	seqs = EntryDB["Sequence"].values
	IDs = EntryDB["BMRB_ID"].values

	seqf = open(seq_fn,'w')

	for i,seq in enumerate(seqs):
		_dump_seq(seqf,seq,IDs[i])

	seqf.close()

	subprocess.run([usearch_exe,
		"-calc_distmx", seq_fn,
		"-tabbedout",distmx_fn,
		"-maxdist","{:4.2f}".format(1.0-similarity)
		])

	subprocess.run([usearch_exe,
		"-cluster_aggd", distmx_fn,
		"-clusterout",clusters_fn,
		"-id","{:4.2f}".format(similarity),
		"-linkage",linkage
		])

	labelsf = open(clusters_fn)
	EntryDB["Cluster_label"] = -1
	for l in labelsf:
		c = l.split()
		eID = int(c[1])
		label = int(c[0])
		EntryDB.loc[EntryDB["BMRB_ID"] == eID,"Cluster_label"] = label

	return EntryDB

def pick_cluster_centers(EntryDB,
	PDB_priority=True,
	reset_index=True):
	# Choose one entry to keep for each cluster
	# Entries with most chemical shifts are prioritized
	# One can choose to give higher priority to having a PDB match (true by default)
	Sort = EntryDB.sort_values(by="N_CS",ascending=False)
	if PDB_priority:
		idx = Sort["PDB_ID"] != "XXXX"
		cat = []
		for i in range(2):
			df = Sort.iloc[np.where(idx)]
			idx = 1 - idx
			cat.append(df)
		Sort = pd.concat(cat)
	labels, idx = np.unique(Sort["Cluster_label"],return_index=True)
	Sort = Sort.iloc[idx]
	if reset_index:
		Sort = Sort.reset_index(drop=True)
	return Sort

def build_CS_database(EntryDB,
	local_files=True,
	NMRSTAR_directory="./NMRSTAR",
	NMRSTAR_prefix="",
	NMRSTAR_suffix=".str",
	tolerance=0.01,
	return_discarded=False,
	sidechain_groups=RESIDUE_SIDECHAIN_GROUPS,
	atom_ignore=RESIDUE_IGNORE_ATOMS):

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
			if pos >= len(seq):
				discarded.append(eID)
				discard = True
				break
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
			if atom_ignore is not None and at in atom_ignore[res_1let]:
				# Ignore specified atoms (not normally observed by NMR)
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


	# Group quasi-equivalent sidechain atoms
	for res, groups in sidechain_groups.items():
		Res_Idx = ( CS_db["Residue"] == res )
		for group, atoms in groups.items():
			at_err = [ "{}_Err".format(at) for at in atoms ]
			group_CS = np.nanmean(CS_db[Res_Idx][atoms],1)
			group_err = np.nanmean(CS_db[Res_Idx][at_err],1)
			CS_db.loc[Res_Idx,group] = group_CS
			CS_db.loc[Res_Idx,group+"_Err"] = group_err

			# Remove individual CS
			CS_db.loc[Res_Idx,atoms] = np.nan
			CS_db.loc[Res_Idx,at_err] = np.nan
		
	# Remove columns with all NaNs
	drop = [ col for col in CS_db.columns if CS_db.dtypes[col] == float and np.all(np.isnan(CS_db[col])) ]
	CS_db = CS_db.drop(columns=drop)

	if return_discarded:
		return CS_db,discarded
	return CS_db

def include_PSIPRED(CS_db,
	PSIPRED_directory="./PSIPRED",
	PSIPRED_prefix="",
	PSIPRED_suffix=".ss2",
	remove_missing=True,
	reset_index=True):
	# This requires previously computed PSIPRED files for each entry
	Q3_names = ["C","H","E"]
	Column_names = ["PSIPRED_"+q for q in Q3_names]
	for c in Column_names:
		CS_db.loc[:,c] = np.nan
	_check_local_dir(PSIPRED_directory)
	d = Path(PSIPRED_directory)
	entries = list(CS_db["BMRB_ID"].value_counts().keys())
	N = len(entries)
	bad_PSIPRED = []
	for i, eID in enumerate(entries):
		fn = d / Path(PSIPRED_prefix+str(eID)+PSIPRED_suffix)
		if not fn.is_file():
			bad_PSIPRED.append(eID)
			continue
		perc = int(i*100/N)
		print("\r  >> Including PSIPRED info: Entry {} ; Progress: {}/{} ({}%)     ".format(eID,i,N,perc), end='')
		Entry_IDX = (CS_db["BMRB_ID"] == eID)
		Entry_CS = CS_db[Entry_IDX]
		ss2f = open(str(fn))
		psipred_seq, psipred_arr = parse_ss2(ss2f)
		ss2f.close()
		seq = "".join(Entry_CS["Residue"].values)
		if seq != psipred_seq:
			bad_PSIPRED.append(eID)
			continue
		CS_db.loc[Entry_IDX,Column_names] = psipred_arr
	if remove_missing:
		for eID in bad_PSIPRED:
			CS_db = CS_db[ CS_db["BMRB_ID"] != eID ]
	if reset_index:
		CS_db = CS_db.reset_index(drop=True)
	return CS_db

def parse_ss2(ss2f):
	ls = ss2f.readlines()
	psipred_arr = np.array([l.split()[3:] for l in ls[2:]])
	seq = "".join([l.split()[1] for l in ls[2:]])
	return seq,psipred_arr

def PSIPRED_seq(seq,fn,psipred_exe="psipred",seqsuf=".fasta"):
	# This requires an installation of blast, PSIPRED, and the uniref90filt database 
	seqfn = "{}{}".format(fn,seqsuf)
	o = open(seqfn,'w')
	_dump_seq(o,seq,fn)
	o.close()
	print("Predicting secondary structure using PSIPRED (may take a few minutes)...")
	subprocess.run([psipred_exe, seqfn])
	return

def run_PSIPRED(EntryDB,directory="./PSIPRED",prefix="",suffix=".ss2",psipred_exe="psipred",skipf=None):
	skip_entries = []
	if skipf is not None:
		skip_entries = set([ l.strip() for l in open(skipf) ])
	d = Path(directory)
	d.mkdir(parents=True, exist_ok=True)
	cwd = os.getcwd()
	os.chdir(str(d))
	for i,entry in EntryDB.iterrows():
		eID = entry["BMRB_ID"]
		seq = entry["Sequence"]
		if str(eID) in skip_entries:
			continue
		check_fn = Path(prefix+str(eID)+suffix)
		if check_fn.is_file():
			continue
		seqfn = prefix+str(eID)
		PSIPRED_seq(seq,seqfn,psipred_exe=psipred_exe)
	os.chdir(cwd)
	return

def check_referencing(CS_db,EntryDB,removesigma=1.5,atoms=["CA","CB","C","H","HA","N"]):
	SS = ["C","H","E"]
	Res = list(CS_db["Residue"].value_counts().keys())
	Mean_CS = np.zeros((len(Res),len(SS),len(atoms)))
	Mean_CS[:] = np.nan
	PSIPRED = np.array([ np.array(CS_db["PSIPRED_{}".format(ss)]) for ss in SS]).T
	PSIPRED_assigned = PSIPRED.argmax(1)
	for iR, r in enumerate(Res):
		Res_IDX = CS_db["Residue"] == r
		X = CS_db[Res_IDX]
		SSMax = PSIPRED_assigned[Res_IDX]
		for iSS, ss in enumerate(SS):
			X_SS = X[SSMax==iSS]
			for iA, at in enumerate(atoms):
				cs = np.nanmean(X_SS[at])
				if np.isnan(cs):
					continue
				Mean_CS[iR,iSS,iA] = cs
	N = len(EntryDB)
	for i, Entry in EntryDB.iterrows():
		eID = Entry["BMRB_ID"]
		perc = int(i*100/N)
		print("\r>> Checking referencing: Entry {} ; {}/{} ({}%)  ".format(eID,i,N,perc), end='')
		Entry_IDX = CS_db["BMRB_ID"] == eID
		Entry_Rows = CS_db[Entry_IDX]
		Entry_CS = np.array(Entry_Rows.loc[:,atoms])
		if not Entry_CS.shape[0]:
			continue
		seq = Entry["Sequence"]
		SSMax = PSIPRED_assigned[Entry_IDX]
		seq_idx = [ Res.index(r) for r in seq ]
		Ref_CS = Mean_CS[seq_idx,SSMax,:]
		diff = np.nanmean(Entry_CS - Ref_CS,axis=0)
		sigma = np.nanstd(Entry_CS,axis=0)
		bad_ref = np.absolute(diff) > sigma*removesigma
		if (np.any(bad_ref)):
			CS_db.loc[Entry_IDX,np.array(atoms)[bad_ref]] = np.nan
	return CS_db

def remove_outliers(CS_db,contamination=0.01):
	atom_nom = _get_PROSECCO_atnames()
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

def cluster_transPRO(CS_db,atoms=["CB","CG","CD"]):
	return _cluster_byCS(CS_db,"P",atoms,"Trans",n_clusters=3,true_idx=2)

def cluster_protHIS(CS_db,atoms=["CA","CB","CD2"]):
	return _cluster_byCS(CS_db,"H",atoms,"Protonated",n_clusters=2,true_idx=1)

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

def keep_entries(Entries,CS_db,recount=True):
	CS_entries = CS_db["BMRB_ID"].values
	valid_entries = Entries["BMRB_ID"].values
	CS_db = CS_db.loc[np.isin(CS_entries,valid_entries)]
	CS_entries = CS_db["BMRB_ID"].values
	if recount:
		N = len(Entries)
		ats = np.array(_get_all_atnames())
		ats = ats[np.isin(ats,CS_db.columns)]
		for i,entry in Entries.iterrows():
			eID = entry["BMRB_ID"]
			perc = int(i*100/N)
			print("\r  >> Keeping entries in both databases: {} / {} ({}%)     ".format(i,N,perc), end='')
			if eID not in CS_entries:
				continue
			Entry_IDX = (CS_entries == eID)
			Entry_CS = np.array(CS_db.loc[Entry_IDX,ats])
			N_CS = np.sum(~np.isnan(Entry_CS))
			Entries.loc[i,"N_CS"] = N_CS
			if not N_CS:
				# Remove entries with no chemical shifts
				CS_db = CS_db.loc[~Entry_IDX]
				Entries = Entries.drop([i])
				CS_entries = CS_db["BMRB_ID"].values
	CS_entries = list(CS_db["BMRB_ID"].value_counts().keys())
	Entries = Entries.loc[np.isin(Entries["BMRB_ID"].values,CS_entries)]
	return Entries, CS_db






