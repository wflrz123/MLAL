import os
from shutil import copyfile
import math

def generate_silence(filepath, dir_new):
    filenames_dir=os.listdir(filepath)
    print(len(filenames_dir),filenames_dir)
    for i in range(0, len(filenames_dir)):
        file_dir = filenames_dir[i]
        print(file_dir)
        print(dir_new+file_dir)
        total_samples = int(596*math.exp(-0.1*i)+4)
        if not os.path.exists(dir_new+file_dir):
            os.mkdir(dir_new+file_dir)
        _, last=os.path.splitext(file_dir)

        filepath_n=filepath+file_dir+"/"
        filenames=os.listdir(filepath_n)

        for j in range(0, total_samples):
            filename = filenames[j]
            name,category=os.path.splitext(filepath_n+filename)
            if category=='.jpg':
                copyfile(filepath_n+filename,dir_new+file_dir+"/"+filename)

                
if __name__ == "__main__":
    filepath = "data/miniImagenet/train/"
    dir_new = "data/miniImagenet_trim/train/"
    generate_silence(filepath, dir_new)