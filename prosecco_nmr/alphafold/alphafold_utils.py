import numpy as np

def generate_domains(target, seq, crop_sizes='64,128,256', crop_step=32):
    windows = [int(x) for x in crop_sizes.split(",")]
    num_residues = len(seq)
    domains = []
    domains.append({"name": target, "description": (1, num_residues)})

    for window in windows:
        starts = list(range(0, num_residues - window, crop_step))
        if num_residues >= window:
            starts += [num_residues - window]
        for start in starts:
            name = f'{target}-l{window}_s{start}'
            domains.append({"name": name, "description": (start + 1, start + window)})
    
    return domains

_CTS = {
    "A" : 35155,
    "C" : 8669,
    "D" : 24161,
    "E" : 28354,
    "F" : 17367,
    "G" : 33229,
    "H" : 9906,
    "I" : 23161,
    "K" : 25872,
    "L" : 40625,
    "M" : 10101,
    "N" : 20212,
    "P" : 23435,
    "Q" : 19208,
    "R" : 23105,
    "S" : 32070,
    "T" : 26311,
    "V" : 29012,
    "W" : 5990,
    "Y" : 14488,
    "X" : 1,
    "-" : 1
}

_tot = 0
for k,v in _CTS.items():
    _tot += v
_FREQ = { k : float(v)/_tot for k, v in _CTS.items()}

AA = 'ACDEFGHIKLMNPQRSTVWYX-'
BACKGROUND_AA_FREQ = np.array([ _FREQ[ch] for ch in AA])
BLOSUM62 = np.load("BLOSUM62.npy")

LAMBDA = 0.3176
QMAT = BACKGROUND_AA_FREQ * np.expand_dims(BACKGROUND_AA_FREQ,-1) * np.exp(LAMBDA*BLOSUM62)
