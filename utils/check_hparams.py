import argparse
from utils import hparams as hp

parser = argparse.ArgumentParser()
parser.add_argument('--hp_file', type=str, default='hparams.py')
args = parser.parse_args()
hp_file = args.hp_file

hp.configure(hp_file)

required_vars = {'':0}
abolished_vars = {'':0}
try_new_vars = {'':0}

for k,v in required_vars:
    pass

for k,v in abolished_vars:
    pass

for k,v in try_new_vars:
    pass