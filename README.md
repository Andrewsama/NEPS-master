# NEPS
Tensorflow implementation of Network Embedding based on High-Degree Penalty
and Adaptive Negative Sampling
## introduction
A tensorflow implementation of NEPS, use random walk of high-degree penalty to get positive pair and adaptive negative sample to get negative pair
## Requirement
python 3.6, tensorflow 1.12.0
## Usage
To run the codes, use:

python main.py

python link_precision.py

## Result
run main.py

you can get the embedding in 'data/'+dataset+'/'+dataset+'_train_features.txt'

run link_precision.py

you can get the result in 'data/'+dataset+'/'+dataset+'_precisionK_result.txt'

