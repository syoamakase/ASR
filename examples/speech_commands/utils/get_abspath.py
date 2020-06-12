import sys
import os

if __name__ == '__main__':
    data_list= sys.stdin.readlines()
    for data in data_list:
        print(os.path.abspath(data.strip()))
