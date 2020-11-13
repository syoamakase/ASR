## CONFIG

train_script = '' # training script
spm_model = '' # sentencepiece model path
test_script = '' # test script
mean_file = None # or a path of mean.npy
var_file = None # or a path of var.npy
save_dir = 'checkpoints' # save path
num_classes = 1000
eos_id = 1
lmfb_dim = 80 # or 80 (English) or 40 (csj)
comment = ''

# When loding
load_checkpoints = False # you must use only train.py if you want to load the saved model
load_checkpoints_path = None # or path
load_checkpoints_epoch = None
is_flatstart = False

# network config
frame_stacking = 3
num_hidden_nodes = 320
num_encoder_layer = 5
encoder_dropout = 0.2
encoder_type = None # 'CNN', 'Wave' None
decoder_type = 'Attention' #'CTC' #or 'Attention' 

# label smooting loss
use_spec_aug = True
# the loss is divided by the length of mel and batch size
T_norm = True
B_norm = False # fix? if True, not converge

# training setting
batch_size = 30
max_epoch = 80
save_per_epoch = 10
lr_adjust_epoch = 20
reset_optimizer_epoch = 40

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
