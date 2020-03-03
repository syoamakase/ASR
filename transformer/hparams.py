
#max_seqlen = 58000
#max_seqlen = 75000
max_seqlen = 200000 #100000 or None
batch_size = None # None or 120
warmup_step = 25000
warmup_factor = 2.0

#better
vocab_size = 1059
lengths_file ='/home/ueno/Datasets/tedlium/lengths.npy'
script_file = '/n/work1/ueno/data/tedlium2/bpe1k/script.bpe.sort_xlen.dataset'
#script_file = '~/Datasets/tedlium/script.bpe1k.sort_xlen.r2.1092'

#vocab_size = 592
#lengths_file ='/home/ueno/Datasets/tedlium/lengths.npy'
#script_file = '/home/ueno/Datasets/tedlium/script.bpe500.sort_xlen.r2.592'

# script_file = '/misc/home/ueno/Datasets/tedlium/script.bpe10k.sort_xlen.r2'
# lengths_file = '/home/ueno/Datasets/tedlium/lengths.npy'

#vocab_size = 6027
#lengths_file ='/home/ueno/Datasets/csj/lengths.npy'
#script_file = '/home/ueno/Datasets/csj/script.bpe.sort_xlen'
