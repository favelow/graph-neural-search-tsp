# Graph Neural Network Guided Local Search for the Traveling Salesperson Problem

This is an implementation of the algorithm described in the paper [Graph Neural Network Guided Local Search for the Traveling Salesperson Problem](https://arxiv.org/abs/2110.05291), owned by favelow.

Want to jump to [the example](https://github.com/favelow/graph-neural-search-tsp#minimal-example)?

## Setup
This project uses various test datasets and models. For correct setup, you need to install [git lfs](https://git-lfs.github.com/).

1. Install [git lfs](https://git-lfs.github.com/)
2. Install [pipenv](https://pipenv.pypa.io)
3. Clone the repo
4. Navigate to the repo and run `pipenv install` in the root directory
5. Run `pipenv shell` to activate the environment

## Datasets
Test datasets used in the paper are in the [data](https://github.com/favelow/graph-neural-search-tsp/tree/master/data) directory.

You can also generate new datasets in two steps: instance generation and preprocessing. You can generate solved TSP instances using:
```
./generate_instances.py <number of instances to generate> <number of nodes> <dataset directory>
```
The specified directory is created. Each instance is a pickled `networkx.Graph`.

Then, prepare the dataset using:
```
./preprocess_dataset.py <dataset directory>
```
This splits the dataset into training, validation, and test sets written to `train.txt`, `val.txt`, and `test.txt` respectively. It also fits a scaler to the training set.

After this step, the datasets can be easily manipulated using `gnngls.TSPDataset`. For example, in [train.py](https://github.com/favelow/graph-neural-search-tsp/blob/master/scripts/train.py#L89).

## Training
Train the model using:
```
./train.py <dataset directory> <tensorboard directory> --use_gpu
```
A new directory will be created under the specified Tensorboard directory, progress and checkpoints will be stored there.

## Testing
Evaluate the model using:
```
./test.py <dataset directory>/test.txt <checkpoint path> <run directory> regret_pred --use_gpu
```
The search progress for all instances in the dataset will be written to the specified run directory as a pickled `pandas.DataFrame`.

For example, you can run the pretraine