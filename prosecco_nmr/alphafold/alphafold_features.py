import numpy as np
import pkg_resources
import subprocess

__all__ = [ 'generate_alphafold_features',
			'run_alphafold'
	]


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