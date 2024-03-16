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
./preprocess_dataset.py <dataset di