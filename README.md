﻿# chess-blunder-predictor

This a Neural Network implementation using Tensorflow aimed to solve the problem of predicting chess blunders given a position and a skill level.

## model.py

This is the main code for training and constructing the model. You will need tensorflow installed for your version of python in order to run it. If you run `python3 model.py --help` then you will see a list of options and hyperparameters that you can set when training the model.

## sagemaker-train.ipynb

This is a sample jupyter notebook for training the model using AWS sagemaker. If you would like to train with AWS, you must first create a notebook instance on AWS and upload this jupyter notebook and model.py to that instance. Then you can just update the hyperparameters accordingly.

## tuner.py

Quick and easy script designed to do some hyperparmeter tuning. Just update the arrays at the top of the files with values that you would like to test and run the script via `python3 tuner.py`. The performance of the models will show up in tensorboard which can be viewed via the command, `tensorboard --logdir "c:\\logs"` on windows.

## spark-jobs

This directory contains the two spark jobs that were written in order to do the data preprocessing.
