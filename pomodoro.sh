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

############## GPU tests ####################
# start GPU reader, that creates a csv file with the GPU energy consumption data, named after the techniques involved
echo "Reading GPU..."
#nvidia-smi --query-gpu=index,timestamp,power.draw,clocks.sm,clocks.mem,clocks.gr --format=csv -l 10 -f ../Data/GPU/GPU_resnet_'$args'.csv &
#echo "Test" >> GPU_resnet_'$args'.csv
# let reader sleep for a while to see a clear turning on of the GPU later
# sleep 31

############## Analysis #####################
# run the analysis script to find a winner
echo "Running GPU results comparison"
#python analysis.py