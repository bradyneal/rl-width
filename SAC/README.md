# Twin Soft Actor Critic

A clean PyTorch implementation of Twin Soft Actor Critic ([SAC](https://arxiv.org/pdf/1812.05905.pdf) + [TD3](https://arxiv.org/abs/1802.09477)). The implementation is based on the [original implementation of TD3](https://github.com/sfujim/TD3).

## Usage
Experiments on single environments can be run by calling:
```bash
python main.py --env HalfCheetah-v2
```

or run SAC with learned temperature

```bash
python main.py --env HalfCheetah-v2 --initial_temperature 0.01 --learn_temperature
```

also you can try to normalize returns that might make training more stable early in training

```bash
python main.py --env HalfCheetah-v2 --initial_temperature 0.01 --learn_temperature --normalize_returns
```

## DeepMind Control Suite

In order to use DeepMind Control Suite, please first install the wrapper from [this repo](https://github.com/1nadequacy/dm_control2gym). And use the camelcase format, for example, DMWalkerWalk-v2.