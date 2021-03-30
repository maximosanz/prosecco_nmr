import numpy as np
import pkg_resources
import subprocess

__all__ = [ 'generate_alphafold_features'
	]

SH_SCRIPT = "gen_alphafold_features.sh"

def generate_alphafold_features(target_ID,**kwargs):
	# Double-dash arguments to the shell script can be passed as keyword arguments
	sh_args = []
	for kw, par in kwargs.items():
		sh_args.extend(['--{}'.format(kw),'{}'.format(par)])

	sh_file = pkg_resources.resource_filename(__name__, SH_SCRIPT)
	cmd = [sh_file] + sh_args
	subprocess.run(cmd)
	return