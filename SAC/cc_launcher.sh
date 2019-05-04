
envname=("HalfCheetah-v1" "Hopper-v1" "Walker2d-v1")

methods=("Baseline") ## dummy - can be anything

seeds=(0 1 2)

width=(20 50 200 400 1200 2000)


for wid_l1 in ${width[@]}
do
  for env in ${envname[@]}
  do
    for seed in ${seeds[@]}
    do
      echo "#!/bin/bash" >> temprun.sh
      echo "#SBATCH --account=rpp-bengioy" >> temprun.sh
      echo "#SBATCH --output=\"/scratch/rislam4/rl_width/SAC/slurm-%j.out\"" >> temprun.sh
      echo "#SBATCH --job-name=rl_wid_sac_env_${env}_method_${method}" >> temprun.sh
      echo "#SBATCH --gres=gpu:1" >> temprun.sh
      echo "#SBATCH --mem=10G" >> temprun.sh
      echo "#SBATCH --time=5:30:00" >> temprun.sh
      echo "module load python/3.6 qt/5.9.6 nixpkgs/16.09 intel/2018.3 cuda/10.0 cudnn/7.5" >> temprun.sh
      echo "source ~/myhome/venvs/mujocoenv/bin/activate" >> temprun.sh
      echo "cd ~/myhome/rl_width/SAC" >> temprun.sh
      echo python main.py --policy_name SAC --env_name ${env[@]} --use_logger --width ${wid_l1[@]} --seed ${seed[@]} --folder ./results/  >> temprun.sh
      eval "sbatch temprun.sh"
      rm temprun.sh
    done
  done
done






