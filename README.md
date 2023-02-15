# Pomodoro

Open topics 10/02
=================

1. Miro Board: create diagram to show training / logging pipeline. [Marlena]
2. Draft overleaf report
3. Get GPU logging running [Max]
4. Debugging Cutmix Mixup [Christian]
5. Commenting + Code Cleaning + Function descriptions
6. analysis.py file to be continued. Requires training logs to do so. Idea for the last lines of the file would be to delete all files on the run folders to ensure a fresh start during consecutive training. [Christian]
7. Loss und Accurarcy Logging [Max]
8. Once step 7 is done, amend shell pipeline.
9. Discuss arguments of shell file.
10. Fix util function subfolder runs [Marlena]
11. Pdf report for best runs and visualisations for winner. Use #borb package for pdf creation. [Marlena]





!!! UTILS NEEDED !!!:
- Util function that creates subfolders per training run "n" in main
- . that logs all paramters configs from main into a separate yaml file in the subfolder run folder (to_yaml(args, path))
- (. that converts the information of the run-config into a PDF via the "borb" package, or imshow (heatmap) plot of the config)
- . that logs all paramters configs + OTHERS (see main.py end) to a csv file in root directory that updates after every run (to_csv(args))
- . that after n runs completes, finds the best configuration via this csv
- . that creates a model with these parameters after all n runs
- . ...

NEW TODO's:
1. Try to mimic sweep visualisations
2. Keep track of the things / work we have done (to hand a summary to Leon in the end). Marlena has started writing notes in our google doc.
3. After energy analysis: main file to write final python file with best sweep configs


(later) 
* Introduce a energy consumption (correlated with number of epochs) / (val) accuracy / inference time quotient or function to be used to measure trained models (but check literature first!)

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
