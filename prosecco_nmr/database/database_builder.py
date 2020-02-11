import requests
from pathlib import Path
import pynmrstar
import warnings
import numpy as np
import pandas as pd
import time
import re
import subprocess

__all__ = [	'get_BMRB_entries',
			'get_NMRSTAR_files',
			'build_entry_database',
			'remove_entries',
			'cluster_sequences',
			'build_CS_database'
			]

__MY_APPLICATION__ = "PROSECCO-NMR"

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

def build_entry_database(	entries,
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
		d = Path(NMRSTAR_directory)
		if not d.is_dir():
			raise ValueError('Cannot build BMRB database: missing NMRSTAR directory {}'.format(NMRSTAR_directory))

	EntryDB = []

	N = len(entries)
	for i, eID in enumerate(entries):

		perc = int(i*100/N)
		print("\r  >> Building database: Entry {} ; Progress: {}/{} ({}%)     ".format(eID,i,N,perc), end='')

		if local_files:
			fn = Path(d / (NMRSTAR_prefix+eID+NMRSTAR_suffix))
			if not fn.is_file():
				warnings.warn("Ignoring BMRB entry {} due to missing file: {}".format(eID,str(fn)))
				continue
			nmrstar = pynmrstar.Entry.from_file(str(fn))
		else:
			nmrstar = pynmrstar.Entry.from_database(eID)

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
		seq = list(seqs.values())[0]

		# Taking only the first PDB match
		pdb = "XXXX"
		if eID in PDBmatch_d:
			pdb = PDBmatch_d[eID][0]

		Entry_d = {	"BMRB_ID" : eID,
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


def remove_entries(	EntryDB,
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


def cluster_sequences(	EntryDB,
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

	subprocess.run([	usearch_exe,
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


def build_CS_database(	EntryDB,
						local_files=True,
						NMRSTAR_directory="./NMRSTAR",
						NMRSTAR_prefix="",
						NMRSTAR_suffix=".str"):

	print("AQUI ESTOY")

	return


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
	seqd = nmrstar.get_tags([	"_Entity.ID",
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


