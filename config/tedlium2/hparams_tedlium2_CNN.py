## CONFIG

train_script = 'examples/tedlium2/train_script.sort_xlen.txt'
spm_model = 'examples/tedlium2/data/train/sentencepice/bpe_1000/model.model'
test_script = 'examples/tedlium2/test_dev.txt'
mean_file = 'examples/tedlium2/data/train/mean.npy'
var_file = 'examples/tedlium2/data/train/var.npy'
lmfb_dim = 80
num_classes = 1000
eos_id = 1

save_dir = 'checkpoints.tedlium2.1kbpe.CNN'

# When loding
load_checkpoints = False # you must use only train.py if you want to load the saved model
load_checkpoints_path = None # or path
load_checkpoints_epoch = None
is_flatstart = False

# network config
frame_stacking = 1 # or False
num_hidden_nodes = 512
num_encoder_layer = 5
encoder_dropout = 0.2
encoder_type = 'CNN' # 'CNN' or None
decoder_type = 'Attention' #'CTC' #or 'Attention' 

# label smooting loss
T_norm = True
B_norm = False # fix? if True, not converge

# training setting
batch_size = 30
max_epoch = 80
save_per_epoch = 10
use_spec_aug = True
lr_adjust_epoch = 20
reset_optimizer_epoch = 40 # None

# inference config (attention)
max_decoder_seq_len = 200
beam_width = 4
shallow_fusion = False
if shallow_fusion:
    LM_path = None

score_func = 'log_softmax' # or 'softmax'

# others
debug_mode = 'tensorboard' # or 'print'
nan_analyze_type = 'ignore' # 'stop'
output_mode = 'other'
