#!/bin/bash

# This script is used to run and control the training pipeline.
# It will train a model for a selected number of runs with randomized optimisation parameters.
# Once training is done, it will analyze the results and find the model seetings
# that used little resources while achieving good classification results.
# For more details, see the README.md file.

############## Main script ##################

#echo "Running baseline..."
#python main.py --epochs 100 --baseline --model resnet50

#echo "Running training..."
#for i in `seq 99`;
#do python main.py --epochs 50 --model resnet50;
#done

############## Analysis #####################
# run the analysis script to find a winner
echo "Running GPU results comparison"
python analysis.py
