import numpy as np
import pandas as pd
import tensorflow as tf
import os

__all__ = [	'make_crops',
			'decode_batch'
	]


# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

def _bytes_feature(value):
	"""Returns a bytes_list from a string / byte."""
	if isinstance(value, type(tf.constant(0))):
		value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
	"""Returns a float_list from a float / double."""
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
	"""Returns an int64_list from a bool / enum / int / uint."""
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _parse_AF_ss2(fn):
	ls = open(fn).readlines()
	return np.array([ l.split()[3:] for l in ls if l.strip() and l.strip()[0] != "#" ] ,dtype=np.float32 )

def _crop_tiling(seqlen,N_crop,Edge,overlap):
	
	N_LargeCrop = N_crop + Edge * 2
	non_overlap = 1.0 - overlap
	stride = int(N_crop*non_overlap)
	last_start = seqlen + Edge - N_crop + stride
	
	starts = np.arange(0,last_start,stride)
	if not len(starts):
		starts = np.array([0])
	starts -= Edge
	ends = starts+ N_LargeCrop
	ends = np.minimum(ends,seqlen)
	starts = np.maximum(starts,0)
	
	tgt_starts = np.zeros(len(starts),dtype=int)
	tgt_starts[0] = Edge
	tgt_ends = ends - starts + tgt_starts
	
	tiling  = np.stack([starts, ends, tgt_starts, tgt_ends]).T
	
	return tiling

def _crop_1Dfeature(X,N_crop,tile):
	src0, src1, tgt0, tgt1 = tile
	X_crop = np.zeros((N_crop,*X.shape[1:]))
	X_crop[tgt0:tgt1] = X[src0:src1]
	return X_crop

def _crop_2Dfeature(X,N_crop,i_tile,j_tile):
	i_src0, i_src1, i_tgt0, i_tgt1 = i_tile
	j_src0, j_src1, j_tgt0, j_tgt1 = j_tile
	X_crop = np.zeros((N_crop,N_crop,*X.shape[2:]))
	X_crop[i_tgt0:i_tgt1,j_tgt0:j_tgt1] = X[i_src0:i_src1,j_src0:j_src1]
	return X_crop

def _get_1D_crops(X,tiling,N_LargeCrop):
	return [ _crop_1Dfeature(X,N_LargeCrop,tile) for tile in tiling ]

def _get_2D_crops(X,tiling,N_LargeCrop):
	N_crops = tiling.shape[0]
	return [ [  _crop_2Dfeature(X,N_LargeCrop,tiling[i_crop],tiling[j_crop]) 
		for j_crop in range(N_crops) ] 
		for i_crop in range(N_crops) ]

def _arr2tfrec(X):
	X = tf.convert_to_tensor(X,dtype=tf.float32)
	X = tf.io.serialize_tensor(X)
	X = _bytes_feature(X)
	return X

DEFAULT_ATOMS = ["CA","CB","C","H","HA","N"]
DEFAULT_SGRAM_BINS = 64
DEFAULT_SPREADS = {"C" : 24.0,
				   "H" : 6.0,
				   "N" : 40.0}

DEFAULT_NCROP = 64

# Standard amino-acid + Cystine (X), Trans proline (O), Protonated histidine (Z)
DEFAULT_AA = "ARNDCQEGHILKMFPSTWYVXOZ"


def _seq_to_onehot(seq,AA=DEFAULT_AA,CS_db=None):
	AA_arr = np.array([c for c in AA])
	seqlen = len(seq)
	seq_arr = np.array([c for c in seq])
	seq_onehot = np.zeros((seqlen,23))
	seq_onehot[np.where(AA_arr == np.expand_dims(seq_arr,-1))] = 1

	# If the Special residue information is encoded in the CS_db
	if CS_db is not None:
		special_res = np.array(CS_db.loc[:,["Cystine","Trans","Protonated"]],dtype=np.float32)
		seq_onehot[:,20:] = special_res
	return seq_onehot


def _make_shiftogram(CS_sec,
					Sgram_NBin=DEFAULT_SGRAM_BINS,
					Sgram_Spreads=DEFAULT_SPREADS,
					ATOMS=DEFAULT_ATOMS):

	seqlen = CS_sec.shape[0]

	Sec_Shiftogram = np.zeros((*CS_sec.shape,Sgram_NBin),dtype=np.float32)

	for attype in Sgram_Spreads:
		at_IDX = np.char.find(np.array(ATOMS), attype) == 0
		NAts = at_IDX.sum()

		at_cs = CS_sec[:,at_IDX]
		nan_IDX = np.isnan(at_cs)

		spread = Sgram_Spreads[attype]
		half_spread = spread / 2
		sec_min = -half_spread
		sec_max = half_spread

		bin_IDX = np.floor( (CS_sec[:,at_IDX] - sec_min) / spread * Sgram_NBin ).astype(int)
		bin_IDX[nan_IDX] = 0
		my_sgram = np.zeros((seqlen,NAts,Sgram_NBin))
		my_sgram[np.arange(seqlen)[:,np.newaxis],np.arange(NAts)[np.newaxis,:],bin_IDX] = 1.0
		my_sgram[nan_IDX] = 0.0
		Sec_Shiftogram[:,at_IDX] = my_sgram

	return Sec_Shiftogram

# Dictionary of features to include in tfrec
# key : (filename_format , parsing_function, NDim)
# filename_format is a string that contains {} which will be replaced by the Entry_ID
# parsing_function is a func that takes in the filename and returns a NumPy array
# NDim is the number of dimensions of the feature

DEFAULT_FEATURE_D = {
	'torsions'   : ("TORSIONS/{}/{}.torsions" , lambda fn: np.load(fn,allow_pickle=True)['probs'] , 1),
	'distogram'  : ("DISTOGRAMS/{}/{}.pickle" , lambda fn: np.load(fn,allow_pickle=True)['probs'] , 2),
	'asa'		 : ("ASAS/{}/{}.ss2" , _parse_AF_ss2 , 1),
	'sec_struct' : ("SECSTRUCTS/{}/{}.ss2" , _parse_AF_ss2 , 1)
}

CROPPING_FUNCTIONS = {
	1 : _get_1D_crops,
	2 : _get_2D_crops
}

DIM_NAMES = ['i','j']

def _format_str(s,substr):
	return s.format(*[substr]*s.count("{}"))


# seqs is a dictionary { entry_ID : seq }
def make_crops(	seqs,
				CS_db=None,
				N_crop=DEFAULT_NCROP,
				Edge=0,
				overlap=0.5,
				Sgram_NBin=DEFAULT_SGRAM_BINS,
				Sgram_Spreads=DEFAULT_SPREADS,
				feat_d=DEFAULT_FEATURE_D,
				ATOMS=DEFAULT_ATOMS,
				AA=DEFAULT_AA,
				RD_SHARDS=0,
				validation_split=0.0,
				training_fn='training',
				validation_fn='validation',
				every=1,
				verbose=True):



	NAtoms = len(ATOMS)
	ATOMS_sec = [ "{}_Sec".format(at) for at in ATOMS ]

	N_LargeCrop = N_crop + Edge*2

	N_Entries = len(seqs)

	if not RD_SHARDS:
		datafile_path = training_fn+'.tfrec'
		writer = tf.io.TFRecordWriter(datafile_path)
	else:
		datafile_paths = [ training_fn+'{}.tfrec'.format(i) for i in range(RD_SHARDS) ]
		writers = [ tf.io.TFRecordWriter(f) for f in datafile_paths]

	if validation_split > 0.0:
		val_datafile_path = validation_fn+'.tfrec'
		val_writer = tf.io.TFRecordWriter(val_datafile_path)

	Crop_CT = 0

	for i, (eID, seq) in enumerate(seqs.items()):
		
		if i % every:
			continue
		
		if verbose:
			print("N_Crops= {} ; Entry {} : {} / {}  ({}%)	".format(Crop_CT,eID,i,N_Entries,int(i*100/N_Entries)),end='\r')

		feats = {}
		for feat, val in feat_d.items():
			fn = _format_str(val[0],eID)
			if not os.path.isfile(fn):
				continue
			# Parse the feature file according to the function in feat_d
			feat_arr = val[1](fn)
			feats[feat] = feat_arr

		seqlen = len(seq)

		CS_sec_MASK = np.ones(CS_sec.shape)
		
		entry_CS = None
		if CS_db is not None:
			entry_CS = CS_db[CS_db["BMRB_ID"] == eID]
			CS_sec = np.array(entry_CS.loc[:,ATOMS_sec])
			nan_sec_IDX = np.isnan(CS_sec)
		
			# Discard entries with no secondary CS
			if np.all(nan_sec_IDX):
				continue

			Sec_Shiftogram = _make_shiftogram(CS_sec,
							Sgram_NBin=DEFAULT_SGRAM_BINS,
							Sgram_Spreads=DEFAULT_SPREADS,
							ATOMS=DEFAULT_ATOMS)

			CS_sec_MASK[nan_sec_IDX] = 0.0
		
		seq_onehot = _seq_to_onehot(seq,CS_db=entry_CS)
		tiling = _crop_tiling(seqlen,N_crop,Edge,overlap)
		N_crops = tiling.shape[0]

		feat_crops = {}
		for feat, val in feat_d.items():
			NDim = val[2]
			feat_crops[feat] = CROPPING_FUNCTIONS[NDim](feats[feat],tiling,N_LargeCrop)

		CS_sec_MASK_crops = _get_1D_crops(CS_sec_MASK,tiling,N_LargeCrop)
		if CS_db is not None:
			Sec_Shiftogram_crops = _get_1D_crops(Sec_Shiftogram,tiling,N_LargeCrop)

		SEQ_onehot_crops = _get_1D_crops(seq_onehot,tiling,N_LargeCrop)
		SEQ_crops = [ seq[tile[0]:tile[1]] for tile in tiling ]

		for i_crop in range(N_crops):
			for j_crop in range(N_crops):
				D_POS = (i_crop,j_crop)
				
				# Skip crops with no secondary CS in the i dimension
				if not np.any(CS_sec_MASK_crops[i_crop] == 1.0):
					continue
				feature = {}

				for feat, val in feat_d.items():
					NDim = val[2]
					# Include long-range i-j crop for 2D features
					if NDim == 2:
						feature['{}{}_{}'.format(*DIM_NAMES,feat)] = _arr2tfrec(feat_crops[feat][i_crop][j_crop])
					# Include 1D features and diagonal 2D features
					for D in range(2):
						arr = feat_crops[feat][D_POS[D]]
						if NDim == 2:
							arr = arr[D_POS[D]]
						feature['{}_{}'.format(DIM_NAMES[D],feat)] = _arr2tfrec(arr)

				# Include res0, sequence and CS information for i, j
				for D in range(2):
					feature['{}_res0'.format(DIM_NAMES[D])] : _int64_feature(tiling[D_POS[D]][0])
					feature['{}_seq_onehot'.format(DIM_NAMES[D])] = _arr2tfrec(SEQ_onehot_crops[D_POS[D]])
					feature['{}_sequence'.format(DIM_NAMES[D])] = _bytes_feature(SEQ_crops[D_POS[D]].encode('utf-8'))
					feature['{}_cs_mask_sec'.format(DIM_NAMES[D])] = _arr2tfrec(CS_sec_MASK_crops[D_POS[D]])

					if CS_db is not None:
						feature['{}_shiftogram_sec'.format(DIM_NAMES[D])] = _arr2tfrec(Sec_Shiftogram_crops[D_POS[D]])

				# Include global information
				feature['entry_ID'] = _int64_feature(eID)
				feature['crop_size'] = _int64_feature(N_crop)
				feature['seqlen'] = _int64_feature(seqlen)

				# Include atomtypes spreads
				for attype, spread in Sgram_Spreads.items():
					feature["{}_spread".format(attype)] : _float_feature(spread)
				
				example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
				serialized_example = example_proto.SerializeToString()
				
				Crop_CT += 1
				
				rd = np.random.uniform()
				if rd < validation_split:
					val_writer.write(serialized_example)
				else:
					if not RD_SHARDS:
						writer.write(serialized_example)
					else:
						rdint = np.random.randint(RD_SHARDS)
						writers[rdint].write(serialized_example)

	if validation_split > 0.0:
		val_writer.close()
	if not RD_SHARDS:
		writer.close()
	else:
		for i in range(RD_SHARDS):
			writers[i].close()
			
	return



i_FEATURES = ['i_seq_onehot','i_torsions','i_sec_struct','i_asa']
j_FEATURES = ['j_seq_onehot','j_torsions','j_sec_struct','j_asa']
ij_FEATURES = ['i_distogram','j_distogram','ij_distogram']
out_FEATURES = ['i_shiftogram_sec']
mask_FEATURES = ['i_cs_mask_sec']

def _tf_parse_batch(egs,feats):
	example = tf.io.parse_example(
		egs, { f : tf.io.FixedLenFeature(shape=(), dtype=tf.string) for f in feats }
	)
	return example

def _prepare_input_batch(examples,
						i_feats=i_FEATURES,
						j_feats=j_FEATURES,
						ij_feats=ij_FEATURES,
						N_Crop=DEFAULT_NCROP,
						Edge=0,
						iEdge_Rd=0,
						jEdge_Rd=0):
	
	iEdge_Rd += Edge
	jEdge_Rd += Edge
	
	arr = []
	for f in i_feats:
		arr.append(tf.map_fn(lambda x: tf.io.parse_tensor(x,tf.float32), examples[f],dtype=tf.float32)[:,iEdge_Rd:iEdge_Rd+N_Crop])
	arr = tf.concat(arr,axis=-1)
	arr_i = tf.expand_dims(arr,-2)
	arr_i = tf.tile(arr_i,[1,1,N_Crop,1])
	
	arr = []
	for f in j_feats:
		arr.append(tf.map_fn(lambda x: tf.io.parse_tensor(x,tf.float32), examples[f],dtype=tf.float32)[:,jEdge_Rd:jEdge_Rd+N_Crop])
	arr = tf.concat(arr,axis=-1)
	arr_j = tf.expand_dims(arr,-3)
	arr_j = tf.tile(arr_j,[1,N_Crop,1,1])
	
	arr_ij = []
	for f in ij_feats:
		arr_ij.append(tf.map_fn(lambda x: tf.io.parse_tensor(x,tf.float32), examples[f],dtype=tf.float32)[:,iEdge_Rd:iEdge_Rd+N_Crop,jEdge_Rd:jEdge_Rd+N_Crop])
	arr_ij = tf.concat(arr_ij,axis=-1)
	
	input_arr = tf.concat([arr_i,arr_j,arr_ij],axis=-1)
	
	return input_arr

def _prepare_output_batch(examples,out_feats=out_FEATURES,N_Crop=DEFAULT_NCROP,Edge=0,Edge_Rd=0):
	Edge_Rd += Edge
	arr = []
	for f in out_feats:
		arr.append(tf.map_fn(lambda x: tf.io.parse_tensor(x,tf.float32), examples[f],dtype=tf.float32)[:,Edge_Rd:Edge_Rd+N_Crop])
	arr = tf.concat(arr,axis=-1)

	return arr

def _prepare_mask_batch(examples,mask_feats=mask_FEATURES,N_Crop=DEFAULT_NCROP,Edge=0,Edge_Rd=0,EPS=10.**-7):
	Edge_Rd += Edge
	arr = []
	for f in mask_feats:
		arr.append(tf.map_fn(lambda x: tf.io.parse_tensor(x,tf.float32), examples[f],dtype=tf.float32)[:,Edge_Rd:Edge_Rd+N_Crop])
	arr = tf.concat(arr,axis=-1)
	arr = tf.expand_dims(arr,-1)
	
	# Epsilon to avoid NaN loss
	arr += EPS
	
	return arr

def decode_batch(egs,
				 i_feats=i_FEATURES,
				 j_feats=j_FEATURES,
				 ij_feats=ij_FEATURES,
				 out_feats=out_FEATURES,
				 mask_feats=mask_FEATURES,
				 N_Crop=DEFAULT_NCROP,
				 Edge=0,
				 Edge_Rd=None):
	feats = i_feats + j_feats + ij_feats + out_feats + mask_feats
	if Edge_Rd is None:
		iEdge_Rd = tf.random.uniform((1,), minval=-Edge, maxval=Edge+1, dtype=tf.dtypes.int32)[0]
		jEdge_Rd = tf.random.uniform((1,), minval=-Edge, maxval=Edge+1, dtype=tf.dtypes.int32)[0]
	examples = _tf_parse_batch(egs,feats)
	in_arr = _prepare_input_batch(examples,i_feats=i_feats,j_feats=j_feats,ij_feats=ij_feats,N_Crop=N_Crop,iEdge_Rd=iEdge_Rd,jEdge_Rd=jEdge_Rd)
	mask_arr = _prepare_mask_batch(examples,mask_feats=mask_feats,N_Crop=N_Crop,Edge_Rd=iEdge_Rd)
	if out_feats:
		out_arr = _prepare_output_batch(examples,out_feats=out_feats,N_Crop=N_Crop,Edge_Rd=iEdge_Rd)
		return (in_arr, mask_arr), out_arr
	return (in_arr, mask_arr)
