import pkg_resources
import numpy as np
from pathlib import Path
from .model import input_fromseq, make_PROSECCO_nn, load_scaler, BACKBONE_ATOMS
from .database import PSIPRED_seq, parse_ss2
from .utils import eval_kwargs

__all__ = ['run',
	'save']

_PSIPRED_SS = ["C","H","E"]


def _predict_singleNet(Input,atoms,**kwargs):
	Net = _load_NN(kwargs['model_name'],atoms)
	CS = Net.predict(Input)
	if kwargs['use_scaler']:
		Res_Scaler = load_scaler(_get_path(kwargs['model_name'],kwargs['scaler_fnpref'],atoms))
		CS = Res_Scaler.inverse_transform(Input,CS)
	return CS


def run(seq,atoms,**kwargs):

	default_kwargs={ 
	'job_name' : "prosecco_job",
	'model_name' : "SLP_base",
	'use_scaler' : True,
	'scaler_fnpref' : "PROSECCO_scaler",
	'return_SS': False,
	'singleAtom_Nets' : True
	}
	kw = eval_kwargs(kwargs,default_kwargs)

	Input = input_fromseq(seq,**kw)
	if kw['return_SS']:
		Input, SS = Input

	if not kw['singleAtom_Nets']:
		CS = _predict_singleNet(Input,atoms,**kw)
	else:
		CS = []
		for atom in atoms:
			atCS = _predict_singleNet(Input,[atom],**kw)
			CS.append(atCS)
		CS = np.concatenate(CS,axis=-1)
		print(CS.shape)

	if kw['return_SS']:
		return CS, SS
	return CS


def save(seq,CS,SS,ofn,fmt='backbone',at_order=BACKBONE_ATOMS):
	o = open(ofn,'w')
	if fmt == "backbone":
		if CS.shape[1] != len(at_order):
			raise ValueError("save() was provided with %d backbone atoms (%d required)" % (CS.shape[1],len(at_order)))
		bb_header = "# ID RES Q3"+ "%8s"*len(at_order) % tuple(at_order) + '\n'
		o.write(bb_header)
		for i,res in enumerate(seq):
			o.write("%4d%4s%3s" % (i+1,
				res,
				_PSIPRED_SS[np.argmax(SS[i])]))
			o.write("%8.3f"*len(at_order) % tuple(CS[i,:]))
			o.write('\n')
	o.close()
	return


def _get_path(model_name,fnpref,atoms):
	pth = Path("model/weights/{}".format(model_name))
	pth = Path(pkg_resources.resource_filename(__name__,str(pth)))
	pth = Path(pth / ( fnpref +
		"".join(["_{}".format(at) for at in atoms])))
	return pth

def _load_NN(model_name,atoms,NN_type="fully_connected",weight_fnpref="PROSECCO_weights"):
	weightspath = _get_path(model_name,weight_fnpref,atoms)
	N_Atoms = len(atoms)
	Net = make_PROSECCO_nn(N_Atoms=N_Atoms)
	Net.load_weights(str(weightspath))
	return Net