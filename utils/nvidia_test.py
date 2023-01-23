import nvidia_ml_py3 as nvml

# Initialize the NVML library
nvml.nvmlInit()

# Get a list of GPU device handles
device_handles = nvml.nvmlDeviceGetHandles()

# Retrieve GPU usage for the first GPU device
gpu_usage = nvml.nvmlDeviceGetUtilizationRates(device_handles[0]).gpu

print("GPU usage:", gpu_usage)

# Close the NVML library
nvml.nvmlShutdown()