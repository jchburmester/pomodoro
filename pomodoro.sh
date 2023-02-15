#!/bin/bash

# This script is used to run the pomodoro technique
# It will train a model for 1 run with randomised optimisation parameters
# It will then have a break and rest
# After a selected number of runs, it will stop the training and learning process
# Then it will analyse the results and find the best performing model
# Then it is happy.

############## Main script ##################
echo "Running training..."
for i in `seq 2`;
do python main.py --epochs 10;
done

############## Analysis #####################
# run the analysis script to find a winner
echo "Running GPU results comparison"
#python analysis.py