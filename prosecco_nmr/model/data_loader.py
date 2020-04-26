import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
import pickle

from .residue_info import RESIDUES, BLOSUM62
from ..database import PSIPRED_seq, parse_ss2

from pathlib import Path

__all__ = ['input_fromseq',
	'make_NN_arrays',
	'make_PROSECCO_nn',
	'make_test_arr',
	'Residue_Scaler',
	'load_scaler'
	]

_Special_Residue_Key = ["Cystine","Trans","Protonated"]
_Special_Residues = ["C","P","H"]
_Special_Residue_chars = ["X","O","Z"]
_PSIPRED_SS = ["C","H","E"]

def _is_outofbounds(pos,NRes,N_neigh):
	isOut = np.zeros(N_neigh*2 + 1,dtype=bool)
	first = pos - N_neigh
	last = pos + N_neigh + 1
	if first < 0:
		isOut[:first*-1] = True
	if last > NRes:
		isOut[-(last-NRes):] = True
	return isOut

def _get_residue_mapping(X,seq_neigh=2,seq_Nodes=23,include_special=True,N_special=3):
	NRes = len(RESIDUES)
	myRes = X.T[seq_Nodes*seq_neigh:seq_Nodes*seq_neigh+NRes].T
	myResIDX = np.argmax(myRes,-1)
	
	if include_special:
		special = X.T[seq_Nodes*seq_neigh+NRes:seq_Nodes*seq_neigh+NRes+N_special].T
		specialIDX = np.arange(NRes,NRes+N_special)
		specialIDX = np.tile(specialIDX,(*special.shape[:-1],1))
		# This would break if single residue is marked as being more than one special case
		# But that should never happen!
		isSpecial = (special == 1)
		anySpecial = np.any(special,axis=-1)
		specialIDX = specialIDX[isSpecial]
		myResIDX[anySpecial] = specialIDX
	return myResIDX

def _get_residue_scaling(X,y,seq_neigh=2,seq_Nodes=23,include_special=True,N_special=3):

	myResIDX = _get_residue_mapping(X,
		seq_neigh=seq_neigh,
		seq_Nodes=seq_Nodes,
		include_special=include_special,
		N_special=N_special)
	NRes = len(RESIDUES)
	resRange = NRes
	if include_special:
		resRange += N_special
	scaling = np.zeros((resRange,2,y.shape[-1]))
	for i in range(resRange):
		cs = y[myResIDX==i]
		scaling[i] = [np.nanmean(cs,0),np.nanstd(cs,0)]
	return scaling, myResIDX

def _extract_PSIPRED_array(entry_CS):
	return np.array([ [ cs["PSIPRED_{}".format(ss)] for ss in _PSIPRED_SS ] 
		for j, cs in entry_CS.iterrows() ])

def _extract_specialRes_info(CS_row):
	return np.array([ CS_row[sp].values[0] for sp in _Special_Residue_Key ],dtype=float)

def _extract_specialRes_array(entry_CS):
	return np.array([ [ cs[sp] for sp in _Special_Residue_Key ] 
		for j, cs in entry_CS.iterrows() ],dtype=float)

def _extract_sequence_array(seq,useBLOSUM=True):
	NRes = len(RESIDUES)
	seqLen = len(seq)
	seqArr = np.zeros((seqLen,NRes))
	for i, r in enumerate(seq):
		try:
			resIDX = RESIDUES.index(r)
			if useBLOSUM:
				seqArr[i] = BLOSUM62[resIDX]
			else:
				seqArr[i,resIDX] = 1.0
		except ValueError:
			seqArr[i] = np.nan
	return seqArr

def _extract_CS_array(entry_CS,atoms):
	return np.array([ [ cs[at] for at in atoms ] 
		for j, cs in entry_CS.iterrows() ])

def _extract_window(pos,arr,N_neigh,add_termini=False):
	window = N_neigh*2 + 1
	window_Input = np.zeros((window,arr.shape[1]))
	isOut = _is_outofbounds(pos,arr.shape[0],N_neigh)
	lIDX = max(0,pos-N_neigh)
	rIDX = min(arr.shape[0],pos+N_neigh+1)
	window_Input[~isOut] = arr[lIDX:rIDX]
	if add_termini:
		add_isOut = np.expand_dims(isOut,-1)
		window_Input = np.concatenate([window_Input,add_isOut.astype(float)],axis=1)
	return window_Input

def _PSIPRED_fromseq(seq,basename,psipred_exe='psipred'):
	PSIPRED_seq(seq,basename,psipred_exe=psipred_exe)
	ss2fn = Path("{}.ss2".format(basename))
	if not ss2fn.is_file():
		raise ValueError("Error running PSIPRED: Output file {} cannot be found".format(ss2fn))
	ss2f = open(ss2fn)
	psipred_seq, psipred_arr = parse_ss2(ss2f)
	if seq != psipred_seq:
		raise ValueError("Error running PSIPRED: Input sequence different to output in file {}".format(ss2fn))
	return psipred_arr

def _specialRes_fromseq(seq):
	seqArr = np.array(list(seq))
	return np.array([ seqArr == ch for ch in _Special_Residue_chars ],dtype=float).T

def input_fromseq(seq,
	basename="prosecco_job",
	NN_type="fully_connected",
	useBLOSUM=True,
	seq_neigh=2,
	SS_neigh=3,
	SS_type="PSIPRED",
	seq_Nodes=23,
	SS_Nodes=4,
	NLP_segment_length=60,
	NLP_segment_stride=5,
	NLP_margins=20,
	psipred_exe='psipred'):
	# NLP parsing not yet implemented here

	# Replace special res with standard AAs
	seq_raw = seq
	for i, ch in enumerate(_Special_Residue_chars):
		std = _Special_Residues[i]
		seq_raw = seq_raw.replace(ch,std)

	seq_Arr = _extract_sequence_array(seq_raw,useBLOSUM=useBLOSUM)
	special_Arr = _specialRes_fromseq(seq)
	# DSSP needs to be implemented
	SS_arr = _PSIPRED_fromseq(seq_raw,basename,psipred_exe=psipred_exe)
	seq_special_Arr = np.concatenate([seq_Arr,special_Arr],axis=1)

	Input = []
	for pos in range(len(seq)):
		seq_Input = _extract_window(pos,seq_special_Arr,seq_neigh)
		SS_Input = _extract_window(pos,SS_arr,SS_neigh,add_termini=True)
		Input.append(np.concatenate([seq_Input.flatten(),SS_Input.flatten()]))

	return np.array(Input)



class Residue_Scaler:
	'''
	Scales the values of chemical shifts (y) by mean/std values of each residue type (encoded in X)
	'''
	def __init__(self,do_std=True,seq_neigh=2,seq_Nodes=23,include_special=True,N_special=3):
		self.scaling = None
		self.seq_neigh = seq_neigh
		self.seq_Nodes = seq_Nodes
		self.include_special = include_special
		self.N_special = N_special
		self.seqRange = len(RESIDUES)
		self.do_std = do_std
		if self.include_special:
			self.seqRange += N_special
		return
	def fit(self,X,y):
		self.scaling, resMap = _get_residue_scaling(X,
			y,
			seq_neigh=self.seq_neigh,
			seq_Nodes=self.seq_Nodes,
			include_special=self.include_special,
			N_special=self.N_special)
		return self

	def _transform_wrapper(self,X,y,f):
		if self.scaling is None:
			raise ValueError('Residue_Scaler: You must use the fit() method before transforming.')
		resMap = _get_residue_mapping(X,
			seq_neigh=self.seq_neigh,
			seq_Nodes=self.seq_Nodes,
			include_special=self.include_special,
			N_special=self.N_special)
		y_new = np.copy(y)
		for i in range(self.seqRange):
			resIDX = (resMap == i)
			cs = y[resIDX]
			new_cs = f(cs,i)
			y_new[resIDX] = new_cs
		return y_new

	def _forward_scale(self,arr,i):
		arr -= self.scaling[i,0]
		if self.do_std:
			arr /= self.scaling[i,1]
		return arr

	def _inverse_scale(self,arr,i):
		if self.do_std:
			arr *= self.scaling[i,1]
		arr += self.scaling[i,0]
		return arr

	def transform(self,X,y):
		y_new = self._transform_wrapper(X,y,self._forward_scale)
		return y_new

	def fit_transform(self,X,y):
		self.fit(X,y)
		return self.transform(X,y)

	def inverse_transform(self,X,y):
		y_new = self._transform_wrapper(X,y,self._inverse_scale)
		return y_new

	def save(self,fn):
		with open(fn, 'wb') as handle:
			pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
		return self

def load_scaler(fn):
	with open(fn, 'rb') as handle:
		b = pickle.load(handle)
	return b

def make_PROSECCO_nn(N_Atoms=1,
	N_Inputs=143,
	N_Hidden=50,
	activation='tanh',
	optimizer='adam',
	loss='mean_squared_error'):
	PROSECCOnet = tf.keras.Sequential()
	PROSECCOnet.add(tf.keras.layers.Dense(N_Hidden, input_dim=N_Inputs, activation=activation))
	PROSECCOnet.add(tf.keras.layers.Dense(N_Atoms))
	PROSECCOnet.compile(loss=loss, optimizer=optimizer)
	return PROSECCOnet

def make_NN_arrays(EntryDB,
	CSdb,
	atoms=["CA","CB","C","HA","H","N"],
	NN_type="fully_connected",
	useBLOSUM=True,
	seq_neigh=2,
	SS_neigh=3,
	SS_type="PSIPRED",
	seq_Nodes=23,
	SS_Nodes=4,
	return_BMRBmap=False,
	return_seqIDX=False,
	ignore_NaN=True,
	NLP_segment_length=60,
	NLP_segment_stride=5,
	NLP_margins=20,
	NLP_discardNaN=0.5):
	'''
	This function makes an input and output array for the neural network from the CSV files.
	The input consists of a window of residues with sequence and secondary structure information.
	For each residue, there is one node for each of the 20 amino acids, plus three extra nodes for
	oxidized CYS, trans PRO and protonated HIS respectively.
	There are four secondary structure nodes: one for Coil, Helix and Sheet and an extra one if the
	residue window is outside the sequence termini.
	You need to specify a list of atoms that you want to extract the CS for (e.g. ["CA"])
	A BMRB map (mapping each row in the array to its BMRB ID) can be returned - helps for entry-based train/test split
	In addition, NLP style arrays can be created, e.g. for LSTM or other seq-to-seq models
	'''

	NSpecial = len(_Special_Residue_Key)
	N_Atoms = len(atoms)

	N = len(EntryDB)
	NRes = len(RESIDUES)
	if SS_type == "PSIPRED":
		SS = _PSIPRED_SS
	NSS = len(SS)

	# Arrays containing the data to train/test the NN
	X = []
	y = []
	if return_BMRBmap:
		BMRB_map = []
	if return_seqIDX:
		seqIDX = []

	margin_col = np.zeros(seq_Nodes+SS_Nodes)
	margin_col[-1] = 1.0

	for i,entry in EntryDB.iterrows():
		eID = entry["BMRB_ID"]
		perc = int(i*100/N)
		print("\r  >> Making NN input: Entry {} ; Progress: {}/{} ({}%)     ".format(eID,i,N,perc), end='')
		seq = entry["Sequence"]
		eCS = CSdb[CSdb["BMRB_ID"]==eID]
		seqLen = len(seq)

		# Need to implement the DSSP parsing here
		if SS_type == "PSIPRED":
			SS_arr = _extract_PSIPRED_array(eCS)

		seq_Arr = _extract_sequence_array(seq,useBLOSUM=useBLOSUM)
		cs_Arr = _extract_CS_array(eCS,atoms)
		special_Arr = _extract_specialRes_array(eCS)

		seq_special_Arr = np.concatenate([seq_Arr,special_Arr],axis=1)

		Input = []
		cs_values = []
		if return_seqIDX:
			seqpos = []

		if NN_type == "fully_connected":
			for pos in range(seqLen):
				pos_cs = cs_Arr[pos]
				if ignore_NaN and np.all(np.isnan(pos_cs)):
					continue
				seq_Input = _extract_window(pos,seq_special_Arr,seq_neigh)
				if np.any(np.isnan(seq_Input)):
					continue
				SS_Input = _extract_window(pos,SS_arr,SS_neigh,add_termini=True)
				if np.any(np.isnan(SS_Input)):
					continue
				cs_values.append(pos_cs)
				Input.append(np.concatenate([seq_Input.flatten(),SS_Input.flatten()]))
				if return_seqIDX:
					seqpos.append(np.array(pos))

		if NN_type == "NLP":
			NaN_frac = np.isnan(cs_Arr).sum(0) / seqLen
			if np.all(NaN_frac > NLP_discardNaN):
				continue
			NLP_Arr = np.concatenate([seq_special_Arr,SS_arr,np.zeros((seqLen,1))],axis=1)
			margin = np.tile(margin_col,(NLP_margins,1))
			NLP_Arr = np.concatenate([margin,NLP_Arr,margin],axis=0)
			cs_margin = np.zeros((NLP_margins,N_Atoms))
			cs_margin[:] = np.nan
			NLP_CS = np.concatenate([cs_margin,cs_Arr,cs_margin],axis=0)
			# Right-padding of sequences shorter than NLP_segment_length:
			if NLP_Arr.shape[0] < NLP_segment_length:
				N_pad = NLP_segment_length-NLP_Arr.shape[0]+1
				rightpad = np.tile(margin_col,(N_pad,1))
				cs_rightpad = np.zeros((N_pad,N_Atoms))
				cs_rightpad[:] = np.nan
				NLP_Arr = np.concatenate([NLP_Arr,rightpad],axis=0)
				NLP_CS = np.concatenate([NLP_CS,cs_rightpad],axis=0)
			for pos in range(0,NLP_Arr.shape[0]-NLP_segment_length,NLP_segment_stride):
				end = pos+NLP_segment_length
				segment_Input = NLP_Arr[pos:end]
				if np.any(np.isnan(segment_Input)):
					continue
				Input.append(segment_Input)
				cs_values.append(NLP_CS[pos:end])
				if return_seqIDX:
					row_seqpos = np.arange(pos-NLP_margins,end-NLP_margins,dtype=np.float64)
					row_seqpos[(row_seqpos < 0) | (row_seqpos >= seqLen)] = np.nan
					seqpos.append(row_seqpos)

		X.extend(Input)
		y.extend(cs_values)
		if return_BMRBmap:
			BMRB_map.extend([eID]*len(Input))
		if return_seqIDX:
			seqIDX.extend(seqpos)

	X = np.array(X)
	y = np.array(y)
	output = [X,y]
	if return_BMRBmap:
		output.append(BMRB_map)
	if return_seqIDX:
		output.append(seqIDX)
	return tuple(output)


def make_test_arr(y,BMRB_map,seqIDX,BMRB_list,maxlen,average_multiple=True):
	'''
	This function makes an array for testing chemical shift predictions consistently
	between different PROSECCO methods and other predictors such as SPARTA+
	Returns an array of shape (N_Entries,max_seq_len,N_Atoms) - right zero-padded
	'''
	test_arr = np.zeros((len(BMRB_list),maxlen,y.shape[-1]))
	test_ct = np.zeros(test_arr.shape)
	for r in range(y.shape[0]):
		y_row = y[r]
		eID = BMRB_map[r]
		if eID not in BMRB_list:
			continue
		row_pos = seqIDX[r]
		row_isNan = np.isnan(row_pos)
		if np.any(row_isNan):
			row_pos = row_pos[~row_isNan]
			y_row = y_row[~row_isNan]
		eIDX = BMRB_list.index(BMRB_map[r])

		cs_isNan = np.isnan(y_row)
		row_pos = row_pos.astype(int)

		this_arr = test_arr[eIDX,row_pos]
		this_ct = test_ct[eIDX,row_pos]

		if not average_multiple:
			this_arr[~cs_isNan] = 0.0
			this_ct[~cs_isNan] = 0
		this_arr[~cs_isNan] += y_row[~cs_isNan]
		this_ct[~cs_isNan] += 1

		test_ct[eIDX,row_pos] = this_ct
		test_arr[eIDX,row_pos] = this_arr
	test_arr /= test_ct
	return test_arr
