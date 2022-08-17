import os
import sys


def replace_names(paths):
    for path in paths:

        for file in os.listdir(path):
            full_path = os.path.join(path, file)
            replace_path = full_path.replace(" ", "_")
            if not os.path.exists(replace_path):
                os.rename(full_path, replace_path)


if __name__=="__main__":
    replace_names(sys.argv[1:])
