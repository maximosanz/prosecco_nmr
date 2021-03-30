import argparse
import itertools
import numpy as np
from pathlib import Path
import scipy.io as sio
from scipy.spatial.distance import pdist, squareform

from .alphafold_utils import generate_domains, BACKGROUND_AA_FREQ, QMAT

def make_crops(seq_file,crop_dir):
    if crop_dir is None:
        crop_dir = seq_file.parent
    else:
        crop_dir = Path(crop_dir)
    target_line, *seq_line = seq_file.read_text().split('\n')
    target = seq_file.stem
    suffix = seq_file.suffix
    target_seq = ''.join(seq_line)

    for domain in generate_domains(target, target_seq):
        name = domain['name']
        if name == target: continue
        crop_start, crop_end = domain["description"]
        seq = target_seq[crop_start-1:crop_end]
        (crop_dir / f'{name}.seq').write_text(f'>{name}\n{seq}')

def get_onehot_aln(aln,aa='ACDEFGHIKLMNPQRSTVWYX-'):
    return (np.expand_dims(aln,-1) == list(aa)).astype(np.float32)

# aas must be along first axis
def shuffle_aa(X,aa_orig,aa_new):
    aa_orig = np.array(list(aa_orig))
    aa_new = np.array(list(aa_new))
    W = np.where(np.expand_dims(aa_new,-1) == aa_orig)[1]
    return X[W]

def sequence_to_onehot(seq):
    aalist = 'ARNDCQEGHILKMFPSTWYVX'
    mapping = {aa: i for i, aa in enumerate(aalist)}
    num_entries = max(mapping.values()) + 1
    one_hot_arr = np.zeros((len(seq), num_entries), dtype=np.float32)

    for aa_index, aa_type in enumerate(seq):
        if aa_type not in aalist:
            aa_type = "X"
        aa_id = mapping[aa_type]
        one_hot_arr[aa_index, aa_id] = 1

    return one_hot_arr

def extract_hmm_profile(hhm_file, sequence, asterisks_replace=0.0):
    """Extracts information from the hmm file and replaces asterisks."""
    profile_part = hhm_file.split('#')[-1]
    profile_part = profile_part.split('\n')
    whole_profile = [i.split() for i in profile_part]
    # This part strips away the header and the footer.
    whole_profile = whole_profile[5:-2]
    gap_profile = np.zeros((len(sequence), 10))
    aa_profile = np.zeros((len(sequence), 20))
    count_aa = 0
    count_gap = 0
    for line_values in whole_profile:
        if len(line_values) == 23:
            # The first and the last values in line_values are metadata, skip them.
            for j, t in enumerate(line_values[2:-1]):
                aa_profile[count_aa, j] = (2**(-float(t) / 1000.) if t != '*' else asterisks_replace)
            count_aa += 1
        elif len(line_values) == 10:
            for j, t in enumerate(line_values):
                gap_profile[count_gap, j] = (2**(-float(t) / 1000.) if t != '*' else asterisks_replace)
            count_gap += 1
        elif not line_values:
            pass
        else:
            raise ValueError('Wrong length of line %s hhm file. Expected 0, 10 or 23'
                           'got %d'%(line_values, len(line_values)))
    hmm_profile = np.hstack([aa_profile, gap_profile])
    assert len(hmm_profile) == len(sequence)
    return hmm_profile

def read_aln(aln_file):
    aln = []
    aln_id = []
    seq = ''
    for line in aln_file.open():
        line = line.strip()
        if line and line[0] == '>':
            aln_id.append(line)
            if seq: aln.append(list(seq))
            seq = ''
        else:
            seq += line
    if seq: aln.append(list(seq))
    aln = np.array(aln)
    return aln, aln_id

def parse_a3m(a3m_fn):
    hhblits_a3m_sequences = []
    myseq = ''
    with open(a3m_fn,'r') as f:
        for l in f:
            if not l.strip():
                continue
            if l[0] != ">":
                myseq += l.strip()
            else:
                if myseq:
                    hhblits_a3m_sequences.append(myseq)
                myseq = ''
        hhblits_a3m_sequences.append(myseq)
    return hhblits_a3m_sequences

def calc_deletion_probability(hhblits_a3m_sequences):
    deletion_matrix = []
    for msa_sequence in hhblits_a3m_sequences:
        deletion_vec = []
        deletion_count = 0
        for j in msa_sequence:
            if j.islower():
                deletion_count += 1
            else:
                deletion_vec.append(deletion_count)
                deletion_count = 0
            deletion_matrix.append(deletion_vec)

    deletion_matrix = np.array(deletion_matrix)
    deletion_matrix[deletion_matrix != 0] = 1.0
    deletion_probability = deletion_matrix.sum(axis=0) / len(deletion_matrix)
    return deletion_probability

def Nc_estimate(aln):
    aln_onehot = get_onehot_aln(aln)
    Nc = np.mean(np.any(aln_onehot,0).sum(-1))
    return Nc

def frobenius_norm_no_gaps(matrix):
    frob = np.linalg.norm(matrix[:,:,:-1,:-1],axis=(2,3))
    frob = (frob + frob.T) * 0.5
    return frob

def add_apc(matrix):
    matrix *= 0.8
    row_mean = (matrix.sum(axis=1) - np.diag(matrix)) / (len(matrix) - 1.0)
    if row_mean.mean() != 0.0:
        return np.outer(row_mean, row_mean) / row_mean.mean()
    else:
        return np.zeros(len(matrix), dtype=np.float32)

def write_aln(aln, aln_id, out_file):
    with out_file.open('w') as f:
        for i in range(len(aln_id)):
            seq = ''.join(aln[i])
            f.write(f'{aln_id[i]}\n{seq}\n')

def sequence_weights(aln):
    cutoff = 0.62 * aln.shape[1]
    similarity = ((np.expand_dims(aln,0) == np.expand_dims(aln,1)).sum(-1)).astype(np.float32)
    similarity = similarity > cutoff
    W = (similarity.sum(0) + similarity.sum(1)) * 0.5
    return 1.0 / W

def feature_generation(seq_file, out_file, crop_dir):
    target_line, *seq_line = seq_file.read_text().split('\n')
    target = seq_file.stem
    target_seq = ''.join(seq_line)
    if crop_dir is None:
        data_dir = seq_file.parent
    else:
        data_dir = Path(crop_dir)
    dataset = []

    for domain in generate_domains(target, target_seq):
        name = domain['name']
        crop_start, crop_end = domain["description"]
        seq = target_seq[crop_start-1:crop_end]
        L = len(seq)
        hhm_file = data_dir / f'{name}.hhm'
        fas_file = data_dir / f'{name}.fas'
        aln_file = data_dir / f'{name}.aln'
        mat_file = data_dir / f'{name}.mat'
        a3m_file = data_dir / f'{name}.a3m'

        if a3m_file.exists():
            hhblits_a3m_sequences = parse_a3m(a3m_file)
            deletion_probability = calc_deletion_probability(hhblits_a3m_sequences).reshape(-1,1)
        else:
            deletion_probability = np.zeros((L, 1), dtype=np.float32)

        if aln_file.exists():
            aln, _ = read_aln(aln_file)
        else:
            aln, aln_id = read_aln(fas_file)
            aln = aln[:, aln[0] != '-']
            write_aln(aln, aln_id, aln_file)
            continue

        if mat_file.exists():
            mat = sio.loadmat(mat_file)
            pseudo_bias = np.float32(mat['pseudo_bias']) * 25  # No idea why...
            pseudolikelihood = np.float32(mat['pseudolikelihood'])
            pseudolikelihood = np.swapaxes(pseudolikelihood,0,1)
            matrix = pseudolikelihood.reshape((L,L,22,22))
            pseudo_frob = frobenius_norm_no_gaps(matrix)
            pseudo_frob -= add_apc(pseudo_frob)
        else:
            pseudo_bias = np.zeros((L, 22), dtype=np.float32)
            pseudo_frob = np.zeros((L, L, 1), dtype=np.float32)
            pseudolikelihood = np.zeros((L, L, 484), dtype=np.float32)

        gap_count = np.float32(aln=='-')
        gap_matrix = np.expand_dims(np.matmul(gap_count.T, gap_count) / aln.shape[0], -1)

        aalist = 'ARNDCQEGHILKMFPSTWYVX-'
        onehot_aln = get_onehot_aln(aln,aalist)
        # Replace unknown amino acids with "X"
        #Xidx = aalist.index("X")
        #onehot_aln[(~np.any(onehot_aln,axis=2)),Xidx] = True

        hhblits_profile = onehot_aln.sum(0).astype(np.float32)
        hhblits_profile /= hhblits_profile.sum(-1).reshape(-1, 1)

        aalist2 = 'ACDEFGHIKLMNPQRSTVWYX-'
        onehot_aln = shuffle_aa(onehot_aln.T,aalist,aalist2).T
        seq_weight = sequence_weights(aln)
        reweighted_profile = (onehot_aln.T * seq_weight).T
        reweighted_profile = reweighted_profile.sum(0).astype(np.float32)
        reweighted_profile /= reweighted_profile.sum(-1).reshape(-1, 1)

        non_gapped_profile = shuffle_aa(hhblits_profile.T,aalist,aalist2[:-1])
        non_gapped_profile /= non_gapped_profile.sum(0)
        non_gapped_profile = non_gapped_profile.T

        Nc = np.mean(np.any(onehot_aln,0).sum(-1))
        ALPHA = Nc - 1
        ALPHA_nogaps = ALPHA - 1
        BETA = 10

        F = onehot_aln.sum(0) / aln.shape[0]
        g = F / BACKGROUND_AA_FREQ
        g = ( np.expand_dims(g,-1) * QMAT ).sum(1)
        # The profile does not sum to one in AlphaFold but closer to 0.85. Not sure why...
        profile_with_prior = (ALPHA*F + BETA*g) / (ALPHA+BETA) * 0.85

        nogaps = np.expand_dims(onehot_aln[:,:,:-1].sum(0).sum(-1),-1)
        F_nogaps = onehot_aln[:,:,:-1].sum(0) / nogaps
        g_nogaps = F_nogaps / BACKGROUND_AA_FREQ[:-1]
        g_nogaps = ( np.expand_dims(g_nogaps,-1) * QMAT[:-1,:-1] ).sum(1)
        profile_with_prior_without_gaps = (ALPHA_nogaps*F_nogaps + BETA*g_nogaps) / (ALPHA_nogaps+BETA)

        data = {
            'chain_name': target,
            'domain_name': name,
            'sequence': seq,
            'seq_length': np.ones((L, 1), dtype=np.int64)*L,
            'residue_index': np.arange(L, dtype=np.int64).reshape(L, 1),
            'aatype': sequence_to_onehot(seq),
            # profile: A profile (probability distribution over amino acid types)
            # computed using PSI-BLAST. Equivalent to the output of ChkParse.
            'hhblits_profile': hhblits_profile,
            'reweighted_profile': reweighted_profile,
            'hmm_profile': extract_hmm_profile(hhm_file.read_text(), seq),
            'num_alignments': np.ones((L, 1), dtype=np.int64) * aln.shape[0],
            'deletion_probability': deletion_probability,
            'gap_matrix': gap_matrix,
            'non_gapped_profile': non_gapped_profile,
            # plmDCA
            'pseudo_frob': pseudo_frob,
            'pseudo_bias': pseudo_bias,
            'pseudolikelihood': pseudolikelihood,
            'num_effective_alignments': np.float32(0.0),
            'mutual_information': np.zeros((L, L, 1), dtype=np.float32),
            # no need features for prediction
            'resolution': np.float32(0),
            'sec_structure': np.zeros((L, 8), dtype=np.int64),
            'sec_structure_mask': np.zeros((L, 1), dtype=np.int64),
            'solv_surf': np.zeros((L, 1), dtype=np.float32),
            'solv_surf_mask': np.zeros((L, 1), dtype=np.int64),
            'alpha_positions': np.zeros((L, 3), dtype=np.float32),
            'alpha_mask': np.zeros((L, 1), dtype=np.int64),
            'beta_positions': np.zeros((L, 3), dtype=np.float32),
            'beta_mask': np.zeros((L, 1), dtype=np.int64),
            'superfamily': '',
            'between_segment_residues': np.zeros((L, 1), dtype=np.int64),
            'phi_angles': np.zeros((L, 1), dtype=np.float32),
            'phi_mask': np.zeros((L, 1), dtype=np.int64),
            'psi_angles': np.zeros((L, 1), dtype=np.float32),
            'psi_mask': np.zeros((L, 1), dtype=np.int64),
            # to be fixed soon
            'profile': np.zeros((L, 21), dtype=np.float32),
            'profile_with_prior': profile_with_prior,
            'profile_with_prior_without_gaps': profile_with_prior_without_gaps
        }
        dataset.append(data)
    
    np.save(out_file, dataset, allow_pickle=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Alphafold - PyTorch version')
    parser.add_argument('-s', '--seq', type=str, required=True, help='target protein fasta file')
    parser.add_argument('-o', '--out', type=str, default=None, help='output file')
    parser.add_argument('-c', '--crop', default=False, action='store_true', help='make crops')
    parser.add_argument('-d', '--crop_directory', default=None, help='directory for crop outputs')
    parser.add_argument('-f', '--feature', default=False, action='store_true', help='make features')
    args = parser.parse_args()

    SEQ_FILE = Path(args.seq)
    if args.crop:
        make_crops(SEQ_FILE,args.crop_directory)
    elif args.feature:
        OUT_FILE = args.out if args.out is not None else SEQ_FILE.parent / SEQ_FILE.stem
        feature_generation(SEQ_FILE, OUT_FILE,args.crop_directory)

