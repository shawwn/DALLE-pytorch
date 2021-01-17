set -ex
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
vae_epoch="${vae_epoch:-30}"

mkdir -p results models

exec python3 -m pdb -c continue trainDALLE.py --dataPath download --vae_epoch "$vae_epoch" "$@"
