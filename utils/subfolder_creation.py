import os

def create_subfolder():
    """
    Makes sure that headfolder 'runs' exists to store information about current run.
    Checks which runs subfolders already exist and creates subsequent one. 
    Subfolders are named by the index of their run.
    """
    # get parent directory
    parent_dir = os.getcwd()

    # create head folder in parent directory to store the subfolders
    head_folder_path = os.path.join(parent_dir, 'runs')

    if not os.path.exists(head_folder_path):
        os.makedirs(head_folder_path)

    # stores all existing runs subfolders
    runs_dirs = sorted(os.listdir(head_folder_path))

    # if first run, creates subfolder with index 0
    if len(runs_dirs) == 0:
        os.makedirs(os.path.join(head_folder_path, str(0)))
        current_dir = str(0)
    
    # make sure that numbers lower 10 get preceded by '0', for sorting reasons
    elif len(runs_dirs) < 10:
        last_index = runs_dirs[-1]
        os.makedirs(os.path.join(head_folder_path, '0' + str(int(last_index)+1)))
        current_dir = '0' + str(int(last_index)+1)

    # if subsequent run, creates subfolder with next index
    else:
        last_index = runs_dirs[-1]
        os.makedirs(os.path.join(head_folder_path, str(int(last_index)+1)))
        current_dir = str(int(last_index)+1)
    
    return current_dir