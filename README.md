# Work2019_DFF_SSM

This repo contains the implementation of "Deep Feature Fusion Model for Sentence Semantic Matching" in Keras & Tensorflow.
# Usage for python code
## 0. Requirement
python 3.6  
numpy==1.16.4  
pandas==0.22.0  
tensorboard==1.12.0  
tensorflow-gpu==1.12.0  
keras==2.2.4  
gensim==3.0.0
## 1. Data preparation
The dataset is Quora & LCQMC.\
"Quora question pairs.", https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs.

"LCQMC: A Large-scale Chinese Question Matching Corpus", https://www.aclweb.org/anthology/C18-1166/.\
## 2. Start the training process
python siamese_NN.py  

## Reference
If you find our source is useful, please consider citing our work.

@article{zhang2019deep,\
  title={Deep Feature Fusion Model for Sentence Semantic Matching},\
  author={Zhang, X and Lu, W and Li, F and Peng, X and Zhang, R},\
  journal={Computers, Materials \& Continua},\
  year={2019},\
  publisher={Computers, Materials and Continua (Tech Science Press)}\
}
