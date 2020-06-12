import argparse

parser = argparse.ArgumentParser()
parser.add_argument('filelist')
args = parser.parse_args()

exception_file1 = 'testing_list.txt'
exception_file2 = 'validation_list.txt'

with open(args.filelist) as f:
    for line in f:
        skip_flag = False
        sp_line = line.strip().split(' ')[0].replace('.wav','').replace('.npy','')

        with open(exception_file1) as f1:
            for line1 in f1:
                line1 = line1.strip().replace('.wav','').replace('.npy','')
                if line1 in sp_line:
                    #print(sp_line, line1)
                    skip_flag = True
                    break

        with open(exception_file2) as f2:
            for line2 in f2:
                line2 = line2.strip().replace('.wav','').replace('.npy','')
                if (line2 in sp_line) or skip_flag:
                    skip_flag = True
                    break

        if not skip_flag:
            print(line.strip())      
        #    pass
        #else:
        #    print(line.strip())      

