# Pomodoro

## Project description

Energy consumption vs. training performance

In this project, we build a streamlined training pipeline to objectively test the energy consumption of training a ResNet-50 model on image classification in different scenarios. We selected three preprocessing techniques and seven model / hyperparameter optimization approaches that are either switched on or off during training.

Four base cases will set the benchmark and mark some bounderies. In the first base case, we apply neither preprocessing nor model optimization ("all-off") and in the second, we apply both ("all-on"). In the third and fourth cases, we switch one of the two on, whereas the other remains switched-off.

The pipeline will return .csv files that contain training and validation accuracy as well as the GPU workload for the different settings.

In the end, we will look at fixed accuracy values to compare the consumed energy. For a second use case, where energy levels are more of interest, we will look at a fixed GPU workload and compare the achieved accuracies. Finally, we will combine the most promising techniques and give our training a final and last shot.

## File structure

## Credits
