import os

def create_subfolder():
    """
    Checks which runs subfolders already exist and creates subsequent one. 
    In new subfolder, store information about current run.
    Subfolders are named by the index of their run.
    """
    # root directory is parent directory
    root_dir = os.path.abspath(os.path.join('.', os.pardir))
    # stores all existing runs subfolders
    runs_dirs = os.listdir(os.path.join(root_dir, 'runs'))

    # if first run, creates subfolder with index 0
    if len(runs_dirs) == 0:
        current_dir = os.makedirs(os.path.join(runs_dirs, '0'))
    # if subsequent run, creates subfolder with next index
    else:
        last_index = runs_dirs[-1]
        current_dir = os.makedirs(os.path.join(root_dir, 'runs', str(int(last_index)+1)))
