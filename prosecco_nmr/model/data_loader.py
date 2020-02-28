import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
from .residue_info import RESIDUES, BLOSUM62

__all__ = ['make_NN_arrays',
	'make_PROSECCO_nn',
	'Residue_Scaler'
	]

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
	myRes = X[:,seq_Nodes*seq_neigh:seq_Nodes*seq_neigh+NRes]
	myResIDX = np.argmax(myRes,1)
	
	if include_special:
		special = X[:,seq_Nodes*seq_neigh+NRes:seq_Nodes*seq_neigh+NRes+N_special]
		specialIDX = np.arange(NRes,NRes+N_special)
		specialIDX = np.tile(specialIDX,(special.shape[0],1))
		# This would break if single residue is marked as being more than one special case
		# But that should never happen!
		isSpecial = (special == 1)
		anySpecial = np.any(special,axis=1)
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
	scaling = np.zeros((resRange,2,y.shape[1]))
	for i in range(resRange):
		cs = y[myResIDX==i]
		scaling[i] = [np.nanmean(cs,0),np.nanstd(cs,0)]
	return scaling, myResIDX

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
	useBLOSUM=True,
	seq_neigh=2,
	SS_neigh=3,
	SS_type="PSIPRED",
	seq_Nodes=23,
	SS_Nodes=4,
	return_BMRBmap=False,
	ignore_NaN=True):
	'''
	This function makes an input and output array for the neural network from the CSV files.
	The input consists of a window of residues with sequence and secondary structure information.
	For each residue, there is one node for each of the 20 amino acids, plus three extra nodes for
	oxidized CYS, trans PRO and protonated HIS respectively.
	There are four secondary structure nodes: one for Coil, Helix and Sheet and an extra one if the
	residue window is outside the sequence termini.
	You need to specify a list of atoms that you want to extract the CS for (e.g. ["CA"])
	A BMRB map (mapping each row in the array to its BMRB ID) can be returned - helps for entry-based train/test split
	'''

	Special_Residue_Key = ["Cystine","Trans","Protonated"]
	Special_Residue_Options = ["C","P","H"]
	NSpecial = len(Special_Residue_Key)

	N = len(EntryDB)
	NRes = len(RESIDUES)
	if SS_type == "PSIPRED":
		SS = ["C","H","E"]
	NSS = len(SS)

	# Arrays containing the data to train/test the NN
	X = []
	y = []
	if return_BMRBmap:
		BMRB_map = []

	for i,entry in EntryDB.iterrows():
		eID = entry["BMRB_ID"]
		perc = int(i*100/N)
		print("\r  >> Making NN input: Entry {} ; Progress: {}/{} ({}%)     ".format(eID,i,N,perc), end='')
		seq = entry["Sequence"]
		eCS = CSdb[CSdb["BMRB_ID"]==eID]
		seqLen = len(seq)

		# Need to implement the DSSP parsing here
		if SS_type == "PSIPRED":
			SS_arr = np.array([ [ cs["PSIPRED_{}".format(ss)] for ss in ["C","H","E"] ] 
				for j, cs in eCS.iterrows() ])
		for j,cs in eCS.iterrows():
			pos = cs["Res_ID"]-1
			res = cs["Residue"]

			cs_values = cs[atoms].values.astype(np.float64)
			if ignore_NaN and np.all(np.isnan(cs_values)):
				continue

			# Check again if it matches - although this has been done at the database building stage
			if res != seq[pos]:
				warnings.warn("Sequence mismatch for entry {} at position {} ({} - {})".format(str(eID),pos+1,res,seq[pos]))
				continue

			# Encode sequence information:
			seq_Window = seq_neigh*2 + 1
			seq_Input = np.zeros((seq_Window,seq_Nodes))
			seq_isOut = _is_outofbounds(pos,seqLen,seq_neigh)
			w0 = pos-seq_neigh
			discard = False
			for iW in range(seq_Window):
				if seq_isOut[iW]:
					continue
				w_res = seq[w0+iW]
				try:
					resIDX = RESIDUES.index(w_res)
				except ValueError:
					discard = True
					break
				if useBLOSUM:
					seq_Input[iW,:NRes] = BLOSUM62[resIDX,:]
				else:
					seq_Input[iW,w_res] = 1.0

				# Special residue info
				if w_res not in Special_Residue_Options:
					continue
				w_cs = eCS[eCS["Res_ID"] == w0+iW + 1]
				special = np.array([ w_cs[sp].values[0] for sp in Special_Residue_Key ],dtype=float)
				seq_Input[iW,-NSpecial:] = special

			if discard:
				continue

			# Encode secondary structure information:
			SS_Window = SS_neigh* 2 + 1
			SS_Input = np.zeros((SS_Window,SS_Nodes))
			SS_isOut = _is_outofbounds(pos,seqLen,SS_neigh)
			w0 = pos-SS_neigh
			myRange = np.arange(w0,w0+SS_Window)[~SS_isOut]
			SS_Input[:,-1] = SS_isOut.astype(float)
			SS_Input[~SS_isOut,:NSS] = SS_arr[myRange]
			if np.any(np.isnan(SS_Input)):
				continue

			# Get neural network input:
			Input = np.concatenate([seq_Input.flatten(),SS_Input.flatten()])
			X.append(Input)
			y.append(cs_values)
			if return_BMRBmap:
				BMRB_map.append(eID)
	X = np.array(X)
	y = np.array(y)
	if return_BMRBmap:
		return X, y, BMRB_map
	return X, y
