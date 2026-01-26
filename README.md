# Message Passing on the Edge: Towards Scalable and Expressive GNNs

Implementation of our paper _Message Passing on the Edge: Towards Scalable and Expressive GNNs_.

<p align="center">
<img src=./EB-GNN.png width=50% height=50%>
</p>


## Setup

Clone this repository and open the directory
```
git clone https://github.com/ocatias/EdgeBasedGNNs; cd EdgeBasedGNNs
```


Add this directory to the python path. 
```
export PYTHONPATH="$PYTHONPATH:$(pwd)"
```

You can reproduce our environment with `envoronment.yml`or `spec-file.txt`. Note that depending on your CUDA version, you might need to adapt the version of pytorch related dependencies.
```
conda env create -f environment.yml
```

Activate environment
```
conda activate EB
```

We use [Umibozu](https://github.com/ocatias/Umibozu) to automate hyperparameter tuning and model evaluation. Set up with (this requires a [Weights and Biases](https://wandb.ai) account):
```
git clone https://github.com/ocatias/Umibozu ; echo '"wandb_project": EdgeBasedGNNs' > Umibozu/config.yaml
```

To use the same Umibozu commit as us:
```
cd Umibozu; git checkout 80444d7547fb9859fc14ccadb93ebaf8976c2080; cd ..
```

## Rerun Experiments

**CSL.** Run with different values of `$seed`:

> python Exp/run_model.py --tracking 0 --dataset CSL --model EBGNN --scheduler Cosine --emb_dim 64 --epochs 1000  --drop_out 0.5 --lr 0.00001 --seed $seed

**BREC.** 
> python Scripts/BREC_eval.py 


**MalNet-Tiny.** Run with different values of `$seed`:

> Exp/run_model.py --dataset malnettiny --emb_dim 64 --pooling mean --num_mp_layers 5 --epochs 500 --scheduler Cosine --lr 0.001 --model EBGNN --batch_size 16 --ff 1 --residual 1 --seed $seed

Experiments on QMD and QM9 use Umibozu for experiment orchestration. Below are the commands to queue the experiments. 

**QMD.**

>  python Umibozu/q.py -code Exp/run_model.py -repeats 5 -metric Final/Val/mae -min True -project EdgeBasedGNNs --*dataset QMD-mulliken_dipole_tot -config Configs/Benchmark/QMD_graph_prediction.yaml

>  python Umibozu/q.py -code Exp/run_model.py -repeats 5 -metric Final/Val/mae -min True -project EdgeBasedGNNs --*dataset QMD-mulliken_quadrupoles -config Configs/Benchmark/QMD_graph_prediction.yaml

>  python Umibozu/q.py -code Exp/run_model.py -repeats 5 -metric Final/Val/mae -min True -project EdgeBasedGNNs --*dataset QMD-EA -config Configs/Benchmark/QMD_graph_prediction.yaml

>  python Umibozu/q.py -code Exp/run_model.py -repeats 5 -metric Final/Val/mae -min True -project EdgeBasedGNNs --*dataset QMD-IP -config Configs/Benchmark/QMD_graph_prediction.yaml

> python Umibozu/q.py -code Exp/run_model.py -repeats 5 -metric Final/Val/mae -min True -project EdgeBasedGNNs --*dataset QMD-bond_length_matrix -config Configs/Benchmark/QMD_edge_prediction.yaml

> python Umibozu/q.py -code Exp/run_model.py -repeats 5 -metric Final/Val/mae -min True -project EdgeBasedGNNs --*dataset  QMD-bond_index_matrix -config Configs/Benchmark/QMD_edge_prediction.yaml

> python Umibozu/q.py -code Exp/run_model.py -repeats 5 -metric Final/Val/mae -min True -project EdgeBasedGNNs --*dataset QMD-natural_ionicity -config Configs/Benchmark/QMD_edge_prediction.yaml

> python Umibozu/q.py -code Exp/run_model.py -repeats 5 -metric Final/Val/mae -min True -project EdgeBasedGNNs --*dataset QMD-mulliken_condensed_charge_matrix -config Configs/Benchmark/QMD_edge_prediction.yaml


**QM9.** For the different QM9 tasks 0 to 11 as dataset (e.g. `--*dataset QM9_0`) run

> python Umibozu/q.py -code Exp/run_model.py -repeats 5 -metric Final/Val/mae -min True -project EdgeBasedGNNs --*dataset QM9_$task -config Configs/Benchmark/QMD_graph_prediction.yaml

**Running queued QMD & QM0 experiments.** After queuing experiments can be run with:
> bash Umibozu/auto.sh

Results can be found in WandB or aggregated using Umibozu (see Umibozu README).
