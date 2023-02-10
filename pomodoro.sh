#!/bin/bash

# The goal of the project is to run energy vs. performance tests of the ResNet-50 model
# We select different pre-processing methods, different optimization techniques for the model,
# and different post-processing steps to find an optimal combination of these techniques for
# little energy consumption with high training performance.

# run the training script with different combinations
echo "Running training with different parameter combinations"
echo "Choose between two models: resnet50, convnextv1. Type --model resnet50 or --model convnextv1 to choose which model to run"

# for i in `seq 10`;do echo hello; done

python main.py

# start GPU reader, that creates a csv file with the GPU energy consumption data, named after the techniques involved
echo "Reading GPU..."
#nvidia-smi --query-gpu=index,timestamp,power.draw,clocks.sm,clocks.mem,clocks.gr --format=csv -l 10 -f ../Data/GPU/GPU_resnet_'$args'.csv &

#echo "Test" >> GPU_resnet_'$args'.csv

# let reader sleep for a while to see a clear turning on of the GPU later
# sleep 31

# run the analysis script to find a winner
echo "Running GPU results comparison"
python analysis.py

done