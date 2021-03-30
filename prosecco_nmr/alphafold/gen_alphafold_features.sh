#!/bin/bash

function show_usage(){
	printf "Usage: $0 [options [parameters]]\n"
	prinft "\n"
	printf "Options:\n"
	printf "-h|--help, Print help\n"
	printf "-t|--target, Target ID\n"
	printf "-o|--outdir, Output directory (default = FEATURES/[target]/ )\n"
	printf "-s|--seq, Sequence file (FASTA, default = FASTA/[target].fasta/ )\n"
	printf "-db|--hhblits_db, HHBlits database location (default = databases/uniclust30_2018_08/uniclust30_2018_08 )\n"
	printf "-ref|--reformat_exe, reformat.pl executable location (default = reformat.pl )\n"
	printf "-hh|--hhblits_exe, hhblits executable location (default = hhblits )\n"
	printf "-psi|--psiblast_exe, hhblits executable location (default = psiblast )\n"
	printf "-oct|--octave_exe, hhblits executable location (default = octave )\n"

return 0
}

# Default values
TARGET=""
TARGET_DIR=""
HHBLITS_DB="databases/uniclust30_2018_08/uniclust30_2018_08"
HHBLITS_EXE="hhblits"
REF_EXE="reformat.pl"
PSIBLAST_EXE="psiblast"
OCTAVE_EXE="octave"

while [ ! -z "$1" ]; do
	if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
		show_usage
	elif [[ "$1" == "--target" ]] || [[ "$1" == "-t" ]]; then
		TARGET="$2"
		shift
	elif [[ "$1" == "--outdir" ]] || [[ "$1" == "-o" ]]; then
		TARGET_DIR="$2"
		shift
	elif [[ "$1" == "--seq" ]] || [[ "$1" == "-s" ]]; then
		TARGET_SEQ="$2"
		shift
	elif [[ "$1" == "--hhblits_db" ]] || [[ "$1" == "-db" ]]; then
		HHBLITS_DB="$2"
		shift
	elif [[ "$1" == "--reformat_exe" ]] || [[ "$1" == "-ref" ]]; then
		REF_EXE="$2"
		shift
	elif [[ "$1" == "--hhblits_exe" ]] || [[ "$1" == "-hh" ]]; then
		HHBLITS_EXE="$2"
		shift
	elif [[ "$1" == "--psiblast_exe" ]] || [[ "$1" == "-psi" ]]; then
		PSIBLAST_EXE="$2"
		shift
	elif [[ "$1" == "--octave_exe" ]] || [[ "$1" == "-oct" ]]; then
		OCTAVE_EXE="$2"
		shift
	fi
shift
done

if [[ "$TARGET" == "" ]]; then 
	printf "Target ID must be specified via --target|-t"
	exit 1
fi
if [[ "$TARGET_DIR" == "" ]]; then 
	TARGET_DIR="FEATURES/${TARGET}"
fi
if [[ "$TARGET_SEQ" == "" ]]; then 
	TARGET_SEQ="FASTA/${TARGET}.fasta"
fi

MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

mkdir -p $TARGET_DIR
cp $TARGET_SEQ ${TARGET_DIR}/${TARGET}.seq

# generate domain crops from target seq
python ${MYDIR}/feature.py -s $TARGET_SEQ -c -d $TARGET_DIR

for domain in ${TARGET_DIR}/*.seq; do
	out=${domain%.seq}
	echo "Generate MSA files for ${out}"
	$HHBLITS_EXE -cpu 4 -i ${out}.seq -d $HHBLITS_DB -oa3m ${out}.a3m -ohhm ${out}.hhm -n 3
	$REF_EXE ${out}.a3m ${out}.fas
	python ${MYDIR}/parse_duplicates.py ${out}.fas ${out}_dup.fas
	NMatches=$(grep \> ${out}_dup.fas | wc -l)
	if [ $NMatches -le 1 ]
	then
	  continue
	fi
	$PSIBLAST_EXE -subject ${out}.seq -in_msa ${out}_dup.fas -out_ascii_pssm ${out}.pssm
	
done

PLMDCA_DIR="${MYDIR}/plmDCA/plmDCA_asymmetric_v2/"
BASE_DIR=$(pwd)

# make target features data and generate ungap target aln file for plmDCA
python ${MYDIR}/feature.py -s $TARGET_SEQ -f -d $TARGET_DIR -o $TARGET_DIR/$TARGET

cd ${PLMDCA_DIR}
for aln in ${BASE_DIR}/${TARGET_DIR}/*.aln; do
	bn=$(echo "$aln" | cut -f 1 -d '.')
	echo "calculate plmDCA for $aln"
	$OCTAVE_EXE --eval plmDCA.m $aln
done
cd -

# run again to update target features data
python ${MYDIR}/feature.py -s $TARGET_SEQ -f -d $TARGET_DIR -o $TARGET_DIR/$TARGET

# Convert to tensorflow format - input to AlphaFold_1

python ${MYDIR}/pickle2tfrec.py ${TARGET_DIR}/${TARGET}.npy ${TARGET_DIR}/${TARGET}.tfrec
