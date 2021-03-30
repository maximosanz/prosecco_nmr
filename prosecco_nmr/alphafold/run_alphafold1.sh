#!/bin/bash
# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


### This is a modified version of the run_eval.sh script in AlphaFold 1

ALPHAFOLD_DIR="alphafold_casp13"
TARGET=""
TFREC_FN=""
OUTPUT_DIR=""
USE_CPU="true"
GPU_ID=""
CONDA_ENV=""
PYTHON_EXE="python3"

function show_usage(){
  printf "Usage: $0 [options [parameters]]\n"
  printf "\n"
  printf "Options:\n"
  printf " -h      | --help,          Print help\n"
  printf " -t      | --target,        Target ID\n"
  printf " -af_dir | --alphafold_dir, Root AlphaFold directory, last dir name must be alphafold_casp13 (default = ${ALPHAFOLD_DIR}/ )\n"
  printf " -o      | --outdir,        Output directory (default = AlphaFold_OUT/[target]/ )\n"
  printf " -tf     | --tfrec,         TFREC input file (default = [target].tfrec/ )\n"
  printf " -g      | --gpu,           ID of GPU to use (default = use CPU )\n"
  printf " -py     | --python_exe,    Python executable (default = python3)\n"

return 0
}

while [ ! -z "$1" ]; do
  if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    show_usage
  elif [[ "$1" == "--alphafold_dir" ]] || [[ "$1" == "-af_dir" ]]; then
    ALPHAFOLD_DIR="$2"
    shift
  elif [[ "$1" == "--target" ]] || [[ "$1" == "-t" ]]; then
    TARGET="$2"
    shift
  elif [[ "$1" == "--tfrec" ]] || [[ "$1" == "-tf" ]]; then
    TFREC_FN="$2"
    shift
  elif [[ "$1" == "--outdir" ]] || [[ "$1" == "-o" ]]; then
    OUTPUT_DIR="$2"
    shift
  elif [[ "$1" == "--gpu" ]] || [[ "$1" == "-g" ]]; then
    GPU_ID="$2"
    USE_CPU="false"
    shift
  elif [[ "$1" == "--env" ]] || [[ "$1" == "-e" ]]; then
    CONDA_ENV="$2"
    shift
  elif [[ "$1" == "--python_exe" ]] || [[ "$1" == "-py" ]]; then
    PYTHON_EXE="$2"
    shift
  fi
shift
done

if [[ "$TARGET" == "" ]]; then 
  printf "Target ID must be specified via --target|-t\n"
  exit 1
fi
if [[ "$TFREC_FN" == "" ]]; then 
  TFREC_FN="${TARGET}.tfrec"
fi
if [[ "$OUTPUT_DIR" == "" ]]; then 
  OUTPUT_DIR="AlphaFold_OUT/${TARGET}"
fi

USE_GPU="--gpu_id=${GPU_ID}"
if [[ "$GPU_ID" == "" ]]; then 
  USE_GPU=''
fi

DISTOGRAM_MODEL="${ALPHAFOLD_DIR}/alphafold-casp13-weights/873731"  # Path to the directory with the distogram model.
BACKGROUND_MODEL="${ALPHAFOLD_DIR}/alphafold-casp13-weights/916425"  # Path to the directory with the background model.
TORSION_MODEL="${ALPHAFOLD_DIR}/alphafold-casp13-weights/941521"  # Path to the directory with the torsion model.

BASE_DIR=$(pwd)
TFREC_FN=$(realpath $TFREC_FN)

mkdir -p "${OUTPUT_DIR}"
echo "Saving output to ${OUTPUT_DIR}/"

cd ${OUTPUT_DIR}
OUTPUT_DIR=$(pwd)

cd ${ALPHAFOLD_DIR}
cd ..

# Run contact prediction over 4 replicas.
for replica in 0 1 2 3; do
  echo "Launching all models for replica ${replica}"

  # Run the distogram model.
  $PYTHON_EXE -m alphafold_casp13.contacts \
    --logtostderr \
    --cpu=${USE_CPU} \
    ${USE_GPU} \
    --config_path="${DISTOGRAM_MODEL}/${replica}/config.json" \
    --checkpoint_path="${DISTOGRAM_MODEL}/${replica}/tf_graph_data/tf_graph_data.ckpt" \
    --output_path="${OUTPUT_DIR}/distogram/${replica}" \
    --eval_sstable="${TFREC_FN}" \
    --stats_file="${DISTOGRAM_MODEL}/stats_train_s35.json" &

wait

  # Run the background model.
  $PYTHON_EXE -m alphafold_casp13.contacts \
    --logtostderr \
    --cpu=${USE_CPU} \
    ${USE_GPU} \
    --config_path="${BACKGROUND_MODEL}/${replica}/config.json" \
    --checkpoint_path="${BACKGROUND_MODEL}/${replica}/tf_graph_data/tf_graph_data.ckpt" \
    --output_path="${OUTPUT_DIR}/background_distogram/${replica}" \
    --eval_sstable="${TFREC_FN}" \
    --stats_file="${BACKGROUND_MODEL}/stats_train_s35.json" &

wait
done
#wait
# Run the torsion model, but only 1 replica.

  $PYTHON_EXE -m alphafold_casp13.contacts \
  --logtostderr \
  --cpu=${USE_CPU} \
  ${USE_GPU} \
  --config_path="${TORSION_MODEL}/0/config.json" \
  --checkpoint_path="${TORSION_MODEL}/0/tf_graph_data/tf_graph_data.ckpt" \
  --output_path="${OUTPUT_DIR}/torsion/0" \
  --eval_sstable="${TFREC_FN}" \
  --stats_file="${TORSION_MODEL}/stats_train_s35.json" 

#echo "All models running, waiting for them to complete"
#wait

echo "Ensembling all replica outputs"

# Run the ensembling jobs for distograms, background distograms.
for output_dir in "${OUTPUT_DIR}/distogram" "${OUTPUT_DIR}/background_distogram"; do
  pickle_dirs="${output_dir}/0/pickle_files/,${output_dir}/1/pickle_files/,${output_dir}/2/pickle_files/,${output_dir}/3/pickle_files/"

  # Ensemble distograms.
  $PYTHON_EXE -m alphafold_casp13.ensemble_contact_maps \
    --logtostderr \
    --pickle_dirs="${pickle_dirs}" \
    --output_dir="${output_dir}/ensemble/"
done

# Only ensemble single replica distogram for torsions.
  $PYTHON_EXE -m alphafold_casp13.ensemble_contact_maps \
  --logtostderr \
  --pickle_dirs="${OUTPUT_DIR}/torsion/0/pickle_files/" \
  --output_dir="${OUTPUT_DIR}/torsion/ensemble/"

echo "Pasting contact maps"

  $PYTHON_EXE -m alphafold_casp13.paste_contact_maps \
  --logtostderr \
  --pickle_input_dir="${OUTPUT_DIR}/distogram/ensemble/" \
  --output_dir="${OUTPUT_DIR}/pasted/" \
  --tfrecord_path="${TFREC_FN}"

cd ${BASE_DIR}

echo "Done"
