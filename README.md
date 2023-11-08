#  README

## Abstract


## Installation
This code use python 3.10.0

To run this program, you'll need to install the following Python packages:
- [jax](https://jax.readthedocs.io)
- [optax](https://github.com/google-deepmind/optax)
- [hydra](https://hydra.cc/)
- [requests](https://pypi.org/project/requests/)
- [matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/)
- [gym](https://pypi.org/project/gym/)

The prefered installation method is by using conda:
```bash
conda env create -f environment.yml
```
## Datasets
- babi-tasks:
automatically downloaded from [](http://www.thespermwhale.com/jaseweston/babi/)
- Omniglot embedding:
We provided omniglot embedding from a standard CNN prototypical network
the data are located at "./datasets/omniglot_proto_emb/"
For our experiments we only use the embedding of omniglot test-set,
meaning none of the classes present in the test-set where used to train the prototypical network.

## Usage
All our tasks relies on hydra configuration system.
### Babi tasks
Single task: (n is the wanted babi task)
```bash
 python run_babi.py task_id=\'n\'
```
Joint training (reproduce setup lsp joint experiment):
```bash
 python run_babi.py task_id=\'1,4,5,6,7,8,9,10,11,12,13,14,15,18,20\' metric=babi_joint
```

### RL tasks
Pair associations:

```bash
python run_rl.py task=ap_omniglot_fixed training=ap_fixed task.generator_params.omniglot_path=./datasets/omniglot_proto_emb/test_set model=ap_pg_fa
```
Match-to-sample (fixed example version):

```bash
python run_rl.py task=mp_omniglot_fixed training=mp_fixed task.generator_params.omniglot_path=./datasets/omniglot_proto_emb/test_set model=mp_pg_fa
```
Match-to-sample (sampled examples version):
```bash
python run_rl.py task=mp_omniglot_sampled training=mp_sampled task.generator_params.omniglot_path=./datasets/omniglot_proto_emb/test_set model=mp_pg_fa
```
Radial maze:

```bash
python run_rl.py task=radial_omniglot_fixed training=radial_fixed task.generator_params.omniglot_path=./datasets/omniglot_proto_emb/test_set model=radial_pg_fa
```
Radial maze (reward switching):
```bash
python run_rl.py task=radial_omniglot_fixed training=radial_fixed task.generator_params.omniglot_path=./datasets/omniglot_proto_emb/test_set model=radial_pg_fa task.switch_reward_rule=everytime
```





