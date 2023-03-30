import os
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set the font size and line width
font_size = 12
line_width = 2

# Initialize lists to store accuracy and loss values
all_accuracies = []
all_losses = []

# Set the loss threshold
loss_threshold = 20

# Loop through all folders in runs/001-100
for i in range(1, 101):
    folder_name = f"runs/{str(i).zfill(3)}"
    
    # Go into each subfolder, find the logs.csv file
    for subfolder in [x for x in os.listdir(folder_name) if x.startswith("logs")]:
        subfolder_path = os.path.join(folder_name, subfolder)

        if os.path.isfile(subfolder_path):
            # Read the logs.csv file and get the accuracy and loss values
            df = pd.read_csv(subfolder_path)
            accuracies = df["accuracy"].tolist()[:-1]  # Remove the last value
            losses = df["loss"].tolist()[1:-1]  # Remove the last value

            if max(losses) < loss_threshold:
                print(f"Run {i}: {len(accuracies)} accuracy values, {len(losses)} loss values")  # Print statement

                # Append the values to the lists
                all_accuracies.append(accuracies)
                all_losses.append(losses)

# Create and save the accuracy plot
plt.figure(figsize=(12, 8))
plt.rcParams.update({"font.size": font_size})
for idx, accuracies in enumerate(all_accuracies):
    plt.plot(accuracies, label=f"Run {idx+1}", linewidth=line_width)
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("report/media/accuracy_plot.png", dpi=300)
plt.show()

# Create and save the loss plot
plt.figure(figsize=(12, 8))
plt.rcParams.update({"font.size": font_size})
for idx, losses in enumerate(all_losses):
    plt.plot(losses, label=f"Run {idx+1}", linewidth=line_width)
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.tight_layout()
plt.savefig("report/media/loss_plot.png", dpi=300)
plt.show()