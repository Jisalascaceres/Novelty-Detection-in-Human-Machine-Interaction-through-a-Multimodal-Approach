# Novelty Detection in Human Machine Interaction through a Multimodal Approach
This repositorie contains the code used in the paper: [Novelty Detection in Human Machine Interaction through a Multimodal Approach](link)

## Installation

To use the code made for the paper, the first thing to do is install the libraries used.
You can use the requirements file provided.
```
    pip install -r requirements.txt
```
## Dataset

To reproduce the experiments, it is necessary is to download the 3 datasets used:

- [X] [AveRobot](http://mozart.dis.ulpgc.es/averobot/)

- [X] [VoxCeleb 1](https://mm.kaist.ac.kr/datasets/voxceleb/#downloads)

- [X] [Mobio](https://www.idiap.ch/en/dataset/mobio)

Once downloaded, put them in folders and generate the embeddings with the correspondent file.

## Experiments

Once the embeddings are generated, the other scripts are free to use.
Every script is named after the experiment it contains.

There are a exception though, "Novelty Pyod" contains examples of many more models than the ones seen in the document.
