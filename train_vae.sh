set -ex
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec python3 -m pdb -c continue trainVAE.py --dataPath download "$@"
