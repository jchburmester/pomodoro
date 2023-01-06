# Pomodoro
NEW TODO's:
1. Write extended config file
2. Try to mimic sweep visualisations
3. Keep track of the things / work we have done (to hand a summary to Leon in the end). Marlena has started writing notes in our google doc.
4. After energy analysis: main file to write final python file with best sweep configs
5. Combinatorics of sweeps: random
    - find default values for optimization techniques for base case (base line)

      
!!! New comparison metrics
- AdamW optimizer (used in almost all SOTA models nowadays, ConvNeXt 2 (https://arxiv.org/pdf/2301.00808.pdf)) available in tensorflow addons
- base learning rate (1.5e-4, 2e-4, 8e-4, 6.25e-3 (> from small to large model size, see convnext2 appendix) 
- weight decay (0.05)
- optimizer momentum (β1 , β2 =0.9, 0.95,    β1 , β2 =0.9, 0.999 (> from small to large model size, see convnext2 appendix))
- learning rate schedule (cosine decay, etc. (look for more))

(later) 
* Make appendix with other models
* Introduce a energy consumption (correlated with number of epochs) / (val) accuracy / inference time quotient or function to be used to measure trained models (but check literature first!)

OLD TODO's: 

* In main.py, convert all current args to a string and pass it to the CSV callback. Because this gets overwritten each run
* Finetune the args in the shell file so that it fits the args in main.py
* Fix the spacing of the first arg in the shell, I have to start it using "python main.py$arg" instead of "python main.py $arg"
* Fix tflite (model quantization)
* Extend custom callback to use nvidia-ml-py3 instead of nvidia-smi in shell. This way the power usage is linked directly to the training and can be exported to the csv file directly (better automation). 
  * Use nvidia-ml-py3 or the wrapper py3nvml (https://py3nvml.readthedocs.io/en/latest/)
  * Information on metrics (like GPU usage, etc.) on https://lambdalabs.com/blog/weights-and-bias-gpu-cpu-utilization
      * This is the same metric that WANDB uses to track training runs

## Project description

Energy consumption vs. training performance

In this project, we build a streamlined training pipeline to objectively test the energy consumption of training a ResNet-50 model on image classification in different scenarios. We selected three preprocessing techniques and seven model / hyperparameter optimization approaches that are either switched on or off during training.

Four base cases will set the benchmark and mark some bounderies. In the first base case, we apply neither preprocessing nor model optimization ("all-off") and in the second, we apply both ("all-on"). In the third and fourth cases, we switch one of the two on, whereas the other remains switched-off.

The pipeline will return .csv files that contain training and validation accuracy as well as the GPU workload for the different settings.

In the end, we will look at fixed accuracy values to compare the consumed energy. For a second use case, where energy levels are more of interest, we will look at a fixed GPU workload and compare the achieved accuracies. Finally, we will combine the most promising techniques and give our training a final and last shot.

## File structure

## Credits
