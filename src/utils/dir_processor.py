import os


# pre create dir for train and test
def create_dir(dir_list):
    for d in dir_list:
        if not os.path.exists(d):
            os.mkdir(d)
