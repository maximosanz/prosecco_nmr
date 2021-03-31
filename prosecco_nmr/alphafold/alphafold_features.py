import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import pkg_resources
import subprocess
import os

__all__ = [ 'generate_alphafold_features',
			'run_alphafold'
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

# Standard amino-acid + Cystine (X), Trans proline (O), Protonated histidine (Z)
DEFAULT_AA = "ARNDCQEGHILKMFPSTWYVXOZ"


def _seq_to_onehot(seq,AA=DEFAULT_AA,CS_db=None):

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
	'torsions'   : ("TORSIONS/{}/{}.torsions" , lambda fn: np.load(fn,allow_pickle=True)['probs'] , 1)
	'distogram'  : ("DISTOGRAMS/{}/{}.pickle" , lambda fn: np.load(fn,allow_pickle=True)['probs'] , 2)
	'asa'		: ("ASAS/{}/{}.ss2" , _parse_AF_ss2 , 1)
	'sec_struct' : ("SECSTRUCTS/{}/{}.ss2" , _parse_AF_ss2 , 1)
}

CROPPING_FUNCTIONS = {
	1 : _get_1D_crops,
	2 : _get_2D_crops
}

DIM_NAMES = ['i','j']

def _format_str(s,substr):
	return s.format(*[substr]*s.count("{}"))

# Currently it requires the CS database - no sequence-only prediction
def make_crops(	Entry_DB,
				CS_db,
				N_crop=64,
				Edge=10,
				overlap=0.5,
				Sgram_NBin=DEFAULT_SGRAM_BINS,
				Sgram_Spreads=DEFAULT_SPREADS,
				feat_d=DEFAULT_FEATURE_D,
				ATOMS=DEFAULT_ATOMS,
				AA=DEFAULT_AA,
				RD_SHARDS=0,
				validation_split=0.1,
				training_fn='training',
				validation_fn='validation',
				every=1,
				verbose=True)



	NAtoms = len(ATOMS)
	ATOMS_sec = [ "{}_Sec".format(at) for at in ATOMS ]

	N_LargeCrop = N_crop + Edge*2

	N_Entries = len(Entries)

	AA_arr = np.array([c for c in AA])

	if not RD_SHARDS:
		datafile_path = training_fn+'.tfrec'
		writer = tf.io.TFRecordWriter(datafile_path)
	else:
		datafile_paths = [ training_fn+'{}.tfrec'.format(i) for i in range(RD_SHARDS) ]
		writers = [ tf.io.TFRecordWriter(f) for f in datafile_paths]


	val_datafile_path = validation_fn+'.tfrec'
	val_writer = tf.io.TFRecordWriter(val_datafile_path)

	Crop_CT = 0

	for i,entry in Entries.iterrows():
		
		if i % every:
			continue
		
		eID = entry["BMRB_ID"]

		if verbose:
			print("N_Crops= {} ; Entry {} : {} / {}  ({}%)	".format(Crop_CT,eID,i,N_Entries,int(i*100/N_Entries)),end='\r')

		feats = {}
		for feat, val in feat_d.items():
			if not os.path.isfile(_format_str(val[0],eID)):
				continue
			# Parse the feature file according to the function in feat_d
			feat_arr = val[1](val[0])
			feats[feat] = feat_arr


		seq = entry["Sequence"]
		seqlen = len(seq)

		entry_CS = CS_db[CS_db["BMRB_ID"] == eID]
		seq_onehot = _seq_to_onehot(seq,CS_db=entry_CS):
		
		CS_sec = np.array(entry_CS.loc[:,ATOMS_sec])
		nan_sec_IDX = np.isnan(CS_sec)
		
		# Discard entries with no secondary CS
		if np.all(nan_sec_IDX):
			continue

		Sec_Shiftogram = _make_shiftogram(CS_sec,
						Sgram_NBin=DEFAULT_SGRAM_BINS,
						Sgram_Spreads=DEFAULT_SPREADS,
						ATOMS=DEFAULT_ATOMS)

		CS_sec_MASK = np.ones(CS_sec.shape)
		CS_sec_MASK[nan_sec_IDX] = 0.0
		
		tiling = crop_tiling(seqlen,N_crop,Edge,overlap)
		N_crops = tiling.shape[0]

		feat_crops = {}
		for feat, val in feat_d.items():
			NDim = val[2]
			feat_crops[feat] = CROPPING_FUNCTIONS[NDim]

		CS_sec_MASK_crops = _get_1D_crops(CS_sec_MASK,tiling,N_LargeCrop)
		Sec_Shiftogram_crops = _get_1D_crops(Sec_Shiftogram,tiling,N_LargeCrop)

		SEQ_onehot_crops = _get_1D_crops(Sec_Shiftogram,tiling,N_LargeCrop)
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
				
	val_writer.close()
	if not RD_SHARDS:
		writer.close()
	else:
		for i in range(RD_SHARDS):
			writers[i].close()
			
	return

def _run_sh(script_name,tID,**kwargs):
	# Double-dash arguments to the shell script can be passed as keyword arguments
	sh_args = []
	for kw, par in kwargs.items():
		sh_args.extend(['--{}'.format(kw),'{}'.format(par)])

	sh_file = pkg_resources.resource_filename(__name__, script_name)
	cmd = ['sh',sh_file,"-t",tID] + sh_args
	subprocess.run(cmd)
	return

FEATURE_SH_SCRIPT = "gen_alphafold_features.sh"

def generate_alphafold_features(target_ID,**kwargs):
	_run_sh(FEATURE_SH_SCRIPT,target_ID,**kwargs)
	return

ALPHAFOLD_SH_SCRIPT = "run_alphafold1.sh"

def run_alphafold(target_ID,**kwargs):
	_run_sh(ALPHAFOLD_SH_SCRIPT,target_ID,**kwargs)
	return





