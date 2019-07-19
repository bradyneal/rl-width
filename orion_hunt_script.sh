#SBATCH --array=1-100%10
#SBATCH --cpus-per-task=8
#SBATCH --output=./parallel-exp.%A.%a.out
#SBATCH --error=./parallel-exp.%A.%a.err
#SBATCH --gres=gpu:1
#SBATCH --job-name=parallel-exp
#SBATCH --mem=10GB
#SBATCH --time=2:59:00

orion -vv hunt -n rl_search --worker-trials 1 --config=orion-config.yml \
  ./hyperparam_main.py --env "CartPole-v1" --algo a2c --log-dir logs/{trial.hash_name} --config ./a2c-orion-cartpole.yml

