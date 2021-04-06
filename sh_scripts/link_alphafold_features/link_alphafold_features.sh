#!/bin/bash

function show_usage(){
	printf "Usage: $0 [options [parameters]]\n"
	printf "\n"
	printf "Options:\n"
	printf " -h   | --help,         Print help\n"
	printf " -t   | --target,       Target ID\n"
	printf " -d   | --feat_dir,     Directory where the raw AlphaFold features are (default = AlphaFold_OUT/[target]/ )\n"

return 0
}

# Default values
TARGET=""
TARGET_DIR=""

while [ ! -z "$1" ]; do
	if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
		show_usage
	elif [[ "$1" == "--target" ]] || [[ "$1" == "-t" ]]; then
		TARGET="$2"
		shift
	elif [[ "$1" == "--feat_dir" ]] || [[ "$1" == "-d" ]]; then
		TARGET_DIR="$2"
		shift
	fi
shift
done

if [[ "$TARGET" == "" ]]; then 
	printf "Target ID must be specified via --target|-t\n"
	exit 1
fi
if [[ "$TARGET_DIR" == "" ]]; then 
	TARGET_DIR="AlphaFold_OUT/${TARGET}"
fi

TARGET_DIR=$(realpath $TARGET_DIR)

mkdir -p TORSIONS/${TARGET}
mkdir -p DISTOGRAMS/${TARGET}
mkdir -p ASAS/${TARGET}
mkdir -p SECSTRUCTS/${TARGET}

ln -s ${TARGET_DIR}/distogram/ensemble/${TARGET}.pickle DISTOGRAMS/${TARGET}
ln -s ${TARGET_DIR}/torsion/0/torsions/${TARGET}.torsions TORSIONS/${TARGET}
ln -s ${TARGET_DIR}/torsion/0/asa/${TARGET}.ss2 ASAS/${TARGET}
ln -s ${TARGET_DIR}/torsion/0/secstruct/${TARGET}.ss2 SECSTRUCTS/${TARGET}
