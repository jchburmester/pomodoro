import os

def create_subfolder():
    """
    Makes sure that headfolder 'runs' exists to store information about current run.
    Checks which runs subfolders already exist and creates subsequent one. 
    Subfolders are named by the index of their run, starting with 001.

    Returns:
    -------
        path_to_current_subfolder
    """
    # get parent directory
    parent_dir = os.getcwd()

    # create head folder in parent directory to store subfolders
    head_folder_path = os.path.join(parent_dir, 'runs')

    os.makedirs(head_folder_path, exist_ok=True)

    # stores all existing runs subfolders
    runs_dirs = sorted(os.listdir(head_folder_path))

    # if first run, creates subfolder with index 001
    if len(runs_dirs) == 0:
        os.makedirs(os.path.join(head_folder_path, '001'))
        current_dir = '001'
    
    # make sure that subfolder names start with '0' and have three digits
    else:
        last_index = runs_dirs[-1]
        last_index_num = int(last_index)
        next_index_num = last_index_num + 1
        next_index_str = str(next_index_num).zfill(3)
        os.makedirs(os.path.join(head_folder_path, next_index_str))
        current_dir = next_index_str
    
    path_to_current_subfolder = os.path.join(head_folder_path, current_dir)
    
    return path_to_current_subfolder