#!/bin/bash

# This script is used to run energy vs. performance tests of the ResNet-50 model
# We select different pre-processing methods, different optimization techniques for the model,
# and different post-processing steps to find an optimal combination of these techniques for
# little energy consumption with high training performance.

# argument name array
declare -a techniques=('augmentation True' 'normalization True' 'new_partitioning True' 'optimizer SGD' 'batch_size 64'
'learning_rate' 'momentum' 'weight_decay' 'global_quantization' 'model_quantization True' 'jit_compilation')

# decision dictionary with booleans as values and techniques as keys
declare -A decision_dictionary
decision_dictionary[all_off]="0 0 0 0 0 0 0 0 0 0 0" # no preprocessing, no optimization
decision_dictionary[all_on]="1 1 1 1 1 1 1 1 1 1 1" # preprocessing and optimization on
decision_dictionary[pre_processing_only]="1 1 1 0 0 0 0 0 0 0 0" # preprocessing on, no optimization
decision_dictionary[model_optimization_only]="0 0 0 1 1 1 1 1 1 1 1" # no preprocessing, all optimization techniques on

decision_dictionary[optimizer]="1 1 1 1 0 0 0 0 0 0 0" # optimizer on
decision_dictionary[batch_size]="1 1 1 0 1 0 0 0 0 0 0" # batch size on
decision_dictionary[learning_rate]="1 1 1 0 0 1 0 0 0 0 0" # learning rate on
decision_dictionary[momentum]="1 1 1 0 0 0 1 0 0 0 0" # momentum on
decision_dictionary[weight_decay]="1 1 1 0 0 0 0 1 0 0 0" # weight decay on
decision_dictionary[global_quantization]="1 1 1 0 0 0 0 0 1 0 0" # global quantization on
decision_dictionary[local_quantization]="1 1 1 0 0 0 0 0 0 1 0" # local quantization on
decision_dictionary[jit_compilation]="1 1 1 0 0 0 0 0 0 0 1" # jit compilation on

# for each key, value pair in the decision dictionary we run the training and energy reader
for key in "${!decision_dictionary[@]}";
do
    # access the decision array/booleans associated with the key
    d="${decision_dictionary[$key]}"

    # for each decision array, we filter the techniques that are turned on
    declare -a args_array=()

    # initialise a counter
    counter=0

    for j in $d; do
        # check if j is 1, if so add the corresponding technique to the args_array
        if [ $j -eq 1 ]; then
            args_array+=("--${techniques[counter]}")
        fi
        # increment the counter
        ((counter++))
    done

    # store the filtered techniques in a string
    args=$(printf " %s" "${args_array[@]}")

    # print the key of the decision array and the filtered techniques
    #echo "$key $args"

    # start GPU reader, that creates a csv file with the GPU energy consumption data, named after the techniques involved
    echo "Reading GPU..."
    #nvidia-smi --query-gpu=index,timestamp,power.draw,clocks.sm,clocks.mem,clocks.gr --format=csv -l 10 -f ../Data/GPU/GPU_resnet_'$args'.csv &

    #echo "Test" >> GPU_resnet_'$args'.csv

    # let reader sleep for a while to see a clear turning on of the GPU later
    # sleep 31

    # run the training script with the filtered techniques
    echo "Running ResNet-50 with the following techniques: $args"
    echo "python main.py$args"
    python main.py$args

done