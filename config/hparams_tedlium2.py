## CONFIG

#train_script = '/n/work1/ueno/data/librispeech/texts/script.word.sort_xlen#'
train_script = '/n/work1/ueno/data/tedlium2/sentencepiece/text.e2e.sort_xlen'
#train_script = '/n/work1/ueno/data/tedlium2/sentencepiece/text.e2e.sort_xlen_sp' 
spm_model = '/n/work1/ueno/data/tedlium2/sentencepiece/models/m.nofisher.model'

save_dir = 'checkpoints.tedlium2.1kbpe'
save_per_epoch = 10

# When loding
load_checkpoints = False # you must use only train.py if you want to load the saved model
load_checkpoints_path = None # or path
load_checkpoints_epoch = None
is_flatstart = False

test_script = '/n/work1/ueno/data/tedlium2/dev_test_htklist'
#mean_file = '/n/work1/ueno/data/tedlium2/sentencepiece/mean_sp.npy'
#var_file = '/n/work1/ueno/data/tedlium2/sentencepiece/var_sp.npy'
mean_file = '/n/work1/ueno/data/tedlium2/sentencepiece/mean.npy'
var_file = '/n/work1/ueno/data/tedlium2/sentencepiece/var.npy'

# general config
lmfb_dim = 80
num_classes = 1000
eos_id = 1

# network config
frame_stacking = 3 # or False
num_hidden_nodes = 320
#num_hidden_nodes_encoder = 512
#num_hidden_nodes_decoder = 1024
num_encoder_layer = 5
encoder_dropout = 0.2
encoder_type = None # 'CNN', 'Wave' None
decoder_type = 'Attention' #'CTC' #or 'Attention' 

# label smooting loss
T_norm = True
B_norm = False # fix? if True, not converge

# training setting
batch_size = 30
max_epoch = 80
use_spec_aug = True
lr_adjust_epoch = 20
reset_optimizer_epoch = 40 # None

# inference config (attention)
max_decoder_seq_len = 200
beam_width = 4
shallow_fusion = False
word_file = None#'/n/rd28/mimura/speech_commands/feature/word.id' #or None
if shallow_fusion:
    LM_path = None

score_func = 'log_softmax' # or 'logit' or 'softmax'

# for previous version
legacy = False

# others
debug_mode = 'print' # or visdom
nan_analyze_type = 'ignore' # 'stop'
output_mode = 'other'
