# Novelty Detection in Human Machine Interaction through a Multimodal Approach
This repositorie contains the code used in the paper: [Novelty Detection in Human Machine Interaction through a Multimodal Approach](https://link.springer.com/chapter/10.1007/978-3-031-49018-7_33)

## Installation

To use the code made for the paper, the first thing to do is install the libraries used.
You can use the requirements file provided.
```
    pip install -r requirements.txt
```


## Embedders

The embedders used in the research were:

- [X] For the audio [X-Vector](https://huggingface.co/pyannote/embedding) (You have to register)
- [X] For the images [FaceNet](https://github.com/faustomorales/keras-facenet)
      
## Dataset

To reproduce the experiments, it is necessary is to download the 3 datasets used:

- [X] [AveRobot](http://mozart.dis.ulpgc.es/averobot/)

- [X] [VoxCeleb 1](https://mm.kaist.ac.kr/datasets/voxceleb/#downloads)

- [X] [Mobio](https://www.idiap.ch/en/dataset/mobio)

Once downloaded, put them in folders and generate the embeddings with the correspondent file. You will probably need to modify the fuctions that load the data.

## Experiments

Once the embeddings are generated, the other scripts are free to use.
Every script is named after the experiment it contains.

There are a exception though, "Novelty Pyod" contains examples of many more models than the ones seen in the document.

In the "Results" folder, you will discover the .csv files containing the performance results obtained from various tests.

For experiments relying on the Genetic Algorithm, it is possible to load both the population and the best individual using Pygad. The files containing this data can be found within the "GA_results" directory.
```
import pygad
    # load the results of the GA
    Results = pygad.load(path)
    # Access to the best individual
    b_ind = Results.best_solution[0]
```

