# Pomodoro

Known bugs:
- global policy float 16 does not work with experimental optimizers (AdamW), maybe works in tf 2.10
- tf optimizer AdamW is not compatible with tf METAL (for MacOS) !! Say this in the report in case they use MacOS. (Does not work with tf-macos 2.9.2, tf-metal 0.5.0)
->> Changed to AdamW from tfa instead of experimental AdamW from tf (seems to be working now), dont know about Macos

- I will remove all the decay params for AdamW. Explain in the report that while this is extremely important for large models, it just doesnt make sense to include in a parameter search that only gets activate when AdamW is chose // and or when using smaller models.
- I will remove post quantization methods. Explain in the paper that the future of green computing probably lies on quantization, but it is in such an early and non-compatible stage that it was not usable in this project, but a very important part for future green energy tranining bla bla.

- Global policy float 16 is not compatible with pre-quantization, so if global policy float 16 is chosen, pre-quantization will be deactivated. (This is not a bug, but a feature, wow)

- jit-compilation does not work on MacOS, METAL (maybe on other systems too).





!!! UTILS NEEDED !!!:
- Util function that creates subfolders per training run "n" in main
- . that logs all paramters configs from main into a separate yaml file in the subfolder run folder (to_yaml(args, path))
- (. that converts the information of the run-config into a PDF via the "borb" package, or imshow (heatmap) plot of the config)
- . that logs all paramters configs + OTHERS (see main.py end) to a csv file in root directory that updates after every run (to_csv(args))
- . that after n runs completes, finds the best configuration via this csv
- . that creates a model with these parameters after all n runs
- . ...




OLD TODO's: 
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
