import pkg_resources
import numpy as np
from pathlib import Path
from .model import input_fromseq, make_PROSECCO_nn, load_scaler, BACKBONE_ATOMS, ALL_ATOMS
from .database import PSIPRED_seq, parse_ss2
from .utils import eval_kwargs

__all__ = ['run',
	'save']

_PSIPRED_SS = ["C","H","E"]


def _predict_singleNet(Input,atoms,**kwargs):
	Net = _load_NN(kwargs['model_name'],atoms)
	if Net is None:
		CS = np.zeros((Input.shape[0],1))
		CS[:] = np.nan
	else:
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
	'singleAtom_Nets' : True,
	'print_progress' : True
	}
	kw = eval_kwargs(kwargs,default_kwargs)

	Input = input_fromseq(seq,**kw)
	if kw['return_SS']:
		Input, SS = Input

	if not kw['singleAtom_Nets']:
		CS = _predict_singleNet(Input,atoms,**kw)
	else:
		CS = []
		for i, atom in enumerate(atoms):
			if kw['print_progress']:
				print("\r  >> Calculating chemical shifts: Progress: {}/{}    ".format(i,len(atoms)), end="")
			atCS = _predict_singleNet(Input,[atom],**kw)
			CS.append(atCS)
		CS = np.concatenate(CS,axis=-1)

	if kw['return_SS']:
		return CS, SS
	return CS


def save(seq,CS,SS,ofn,fmt='backbone',at_order=None):
	o = open(ofn,'w')
	if at_order is None:
		if fmt == "backbone":
			at_order = BACKBONE_ATOMS
		elif fmt == "sidechain":
			at_order = ALL_ATOMS
	if at_order is None or CS.shape[1] != len(at_order):
		raise ValueError("save() was provided with %d atoms (%d expected)" % (CS.shape[1],len(at_order)))
	N_Atoms = len(at_order)

	if fmt == "backbone":
		bb_header = "# ID RES Q3"+ "%8s"*len(at_order) % tuple(at_order) + '\n'
		o.write(bb_header)
		for i,res in enumerate(seq):
			o.write("%4d%4s%3s" % (i+1,
				res,
				_PSIPRED_SS[np.argmax(SS[i])]))
			o.write("%8.3f"*len(at_order) % tuple(CS[i,:]))
			o.write('\n')

	elif fmt == "sidechain":
		sd_header = "# ID RES Q3     ATOM       CS \n"
		for i,res in enumerate(seq):
			for j in range(N_Atoms):
				cs = CS[i,j]
				if np.isnan(cs):
					continue
				o.write("%4d%4s%3s%9s %8.3f\n" % (i+1,
					res,
					_PSIPRED_SS[np.argmax(SS[i])],
					at_order[j],
					cs))


		pass
	o.close()
	return


def _get_path(model_name,fnpref,atoms,return_fn=False):
	pth = Path("model/weights/{}".format(model_name))
	pth = Path(pkg_resources.resource_filename(__name__,str(pth)))
	fn = ( fnpref +
		"".join(["_{}".format(at) for at in atoms]))
	pth = Path(pth / fn )
	if return_fn:
		return pth, fn
	return pth

def _load_NN(model_name,atoms,NN_type="fully_connected",weight_fnpref="PROSECCO_weights"):
	weightspath, w_fn = _get_path(model_name,weight_fnpref,atoms,return_fn=True)
	wFiles = list(weightspath.parents[0].glob('{}*'.format(w_fn)))
	if not wFiles:
		return None
	N_Atoms = len(atoms)
	Net = make_PROSECCO_nn(N_Atoms=N_Atoms)
	Net.load_weights(str(weightspath))
	return Net