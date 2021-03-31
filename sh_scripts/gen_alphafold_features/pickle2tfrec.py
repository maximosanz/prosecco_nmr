import json
import pickle
import collections
import numpy as np
import tensorflow as tf



NUM_RES = None
FEATURES = {
    'aatype': ('float32', [NUM_RES, 21]),
    'alpha_mask': ('int64', [NUM_RES, 1]),
    'alpha_positions': ('float32', [NUM_RES, 3]),
    'beta_mask': ('int64', [NUM_RES, 1]),
    'beta_positions': ('float32', [NUM_RES, 3]),
    'between_segment_residues': ('int64', [NUM_RES, 1]),
    'chain_name': ('string', [1]),
    'deletion_probability': ('float32', [NUM_RES, 1]),
    'domain_name': ('string', [1]),
    'gap_matrix': ('float32', [NUM_RES, NUM_RES, 1]),
    'hhblits_profile': ('float32', [NUM_RES, 22]),
    'hmm_profile': ('float32', [NUM_RES, 30]),
    'key': ('string', [1]),
    'mutual_information': ('float32', [NUM_RES, NUM_RES, 1]),
    'non_gapped_profile': ('float32', [NUM_RES, 21]),
    'num_alignments': ('int64', [NUM_RES, 1]),
    'num_effective_alignments': ('float32', [1]),
    'phi_angles': ('float32', [NUM_RES, 1]),
    'phi_mask': ('int64', [NUM_RES, 1]),
    'profile': ('float32', [NUM_RES, 21]),
    'profile_with_prior': ('float32', [NUM_RES, 22]),
    'profile_with_prior_without_gaps': ('float32', [NUM_RES, 21]),
    'pseudo_bias': ('float32', [NUM_RES, 22]),
    'pseudo_frob': ('float32', [NUM_RES, NUM_RES, 1]),
    'pseudolikelihood': ('float32', [NUM_RES, NUM_RES, 484]),
    'psi_angles': ('float32', [NUM_RES, 1]),
    'psi_mask': ('int64', [NUM_RES, 1]),
    'residue_index': ('int64', [NUM_RES, 1]),
    'resolution': ('float32', [1]),
    'reweighted_profile': ('float32', [NUM_RES, 22]),
    'sec_structure': ('int64', [NUM_RES, 8]),
    'sec_structure_mask': ('int64', [NUM_RES, 1]),
    'seq_length': ('int64', [NUM_RES, 1]),
    'sequence': ('string', [1]),
    'solv_surf': ('float32', [NUM_RES, 1]),
    'solv_surf_mask': ('int64', [NUM_RES, 1]),
    'superfamily': ('string', [1]),
}
Protein = collections.namedtuple('Protein', ['len', 'seq', 'inputs_1d', 'inputs_2d', 'inputs_2d_diagonal', 'scalars', 'targets'])

def tfrec_read(tfrec_file):
    
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    features = [
        'aatype',
        'beta_mask',
        'beta_positions',
        'between_segment_residues',
        'chain_name',
        'deletion_probability',
        'domain_name',
        'gap_matrix',
        'hhblits_profile',
        'hmm_profile',
        'non_gapped_profile',
        'num_alignments',
        'num_effective_alignments',
        'profile',
        'profile_with_prior',
        'profile_with_prior_without_gaps',
        'pseudo_bias',
        'pseudo_frob',
        'pseudolikelihood',
        'residue_index',
        'resolution',
        'reweighted_profile',
        'sec_structure',
        'sec_structure_mask',
        'seq_length',
        'sequence',
        'solv_surf',
        'solv_surf_mask',
        'superfamily'
    ]
    features = {name: FEATURES[name] for name in features}

    def parse_tfexample(raw_data, features):
        feature_map = {k: tf.io.FixedLenSequenceFeature(shape=(), dtype=eval(f'tf.{v[0]}'), allow_missing=True) for k, v in features.items()}
        parsed_features = tf.io.parse_single_example(raw_data, feature_map)
        num_residues = tf.cast(parsed_features['seq_length'][0], dtype=tf.int32)

        for k, v in parsed_features.items():
            new_shape = [num_residues if s is None else s for s in FEATURES[k][1]]
            assert_non_empty = tf.assert_greater(tf.size(v), 0, name=f'assert_{k}_non_empty',
                message=f'The feature {k} is not set in the tf.Example. Either do not '
                'request the feature or use a tf.Example that has the feature set.')
            with tf.control_dependencies([assert_non_empty]):
                parsed_features[k] = tf.reshape(v, new_shape, name=f'reshape_{k}')
        return parsed_features

    raw_dataset = tf.data.TFRecordDataset([tfrec_file])
    raw_dataset = raw_dataset.map(lambda raw: parse_tfexample(raw, features))
    return raw_dataset

def tfrec2pkl(dataset, pkl_file):
    datalist = []
    dataset = dataset.batch(1)
    for x in dataset:
        data = {}
        for k, v in x.items():
            if k in ['sequence', 'domain_name', 'chain_name', 'resolution', 'superfamily', 'num_effective_alignments']:
                if v.numpy().dtype == 'O':
                    data[k] = v.numpy()[0,0].decode('utf-8')
                else:
                    data[k] = v.numpy()[0,0]
            else:
                data[k] = v.numpy()[0]
        datalist.append(data)

    with open(pkl_file, 'wb') as f:
        pickle.dump(datalist, f)

    return datalist

import sys
data_file = sys.argv[1]
out_filename = sys.argv[2]
dataset_pickle = np.load(data_file, allow_pickle=True)


# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

FMAP = {
	'int64' : _int64_feature,
	'float32' : _float_feature,
	'string' : _bytes_feature
}

def serialize_example(feat_d):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    seq_length = feat_d['seq_length'][0][0]
    feature = {}
    for feat, val in FEATURES.items():
        if feat not in feat_d:
            if val[0] == 'string':
                T = ''
            else:
                shape = tuple([ x if x is not None else seq_length for x in val[1] ])
                T = np.zeros(shape,dtype=val[0])
        else:
            T = feat_d[feat]
        if feat == 'sequence':
            T = ''.join(T)
        f = FMAP[val[0]]
        
        if type(T) == np.ndarray:
            T = T.flatten()
            T = list(T)
        elif type(T) == str:
        	T = [str.encode(T)]

        if type(T) != list:
            T = [T]
        feature[feat] = f(T)
  
    # Create a Features message using tf.train.Example.
  
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

with tf.io.TFRecordWriter(out_filename) as writer:
    for entry in dataset_pickle:
        example = serialize_example(entry)
        writer.write(example)
