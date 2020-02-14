## Path Ranking with Attention to Type Hierarchies
This repo contains code for training and testing the proposed models in *Path Ranking with Attention to Type Hierarchies*.
Due to its large size, data needs to be downloaded separately from [dropbox](https://www.dropbox.com/s/0a4o2jljg4imuux/data.zip?dl=0).

## Notes
1. Code for baseline models in the paper can be found [here](https://github.com/matt-gardner/pra) (PRA and SFE) and 
[here](https://github.com/rajarshd/ChainsofReasoning) (Path-RNN).
2. We provide tokenized data for WN18RR and FB15k-237. Our data format follows 
[*ChainsofReasoning*](https://github.com/rajarshd/ChainsofReasoning). Vocabularies used for tokenizing data are also
provided for reference.
3. Raw data for WN18RR and FB15k-237 can be found 
[here](https://github.com/TimDettmers/ConvE). Types for WN18RR entities can be obtained from Wordnet. Types for 
FB15k-237 entities can be found [here](https://github.com/thunlp/TKRL).

## Tested platform
* Hardware: 64GB RAM, 12GB GPU memory
* Software: ubuntu 16.04, python 3.5, cuda 8

## Setup
1. Install cuda
2. (Optional) Set up python virtual environment by running `virtualenv -p python3 .`
3. (Optional) Activate virtual environment by running `source bin/activate`
3. Install pytorch with cuda
4. Install requirements by running `pip3 install -r requirements.txt`

## Instruction for running the code
### Data
1. Compressed data file can be downloaded from [dropbox](https://www.dropbox.com/s/0a4o2jljg4imuux/data.zip?dl=0)
2. Unzip the file in the root directory of this repo.

### Run the model
1. Use `run.py` to train and test the model on WN18RR or FB15k-237.
2. Use `/main/playground/model2/CompositionalVectorSpaceAlgorithm.py` to modify the training settings and hyperparamters.
3. Use `main/playground/model2/CompositionalVectorSpaceModel.py` to modify the network design. Different attention methods for
types and paths can be selected here.
4. Training progress can be monitored using tensorboardX by running `tensorboard --logdir runs`. Tutorials and Details can be found [here](https://github.com/lanpa/tensorboardX).
