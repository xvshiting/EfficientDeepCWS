import os
def init_dir(dir_path):
    if os.path.exists(dir_path):
        print("{} exist!".format(dir_path))
    else:
        os.makedirs(dir_path)
        print("Build {} success!".format(dir_path))

def delete_file(file):
    if os.path.exists(file):
        os.remove(file)

