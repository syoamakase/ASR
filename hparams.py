## CONFIG

train_script = '/n/work1/ueno/data/librispeech/texts/script.word.sort_xlen'
save_dir = 'checkpoints.word.libri960'
load_checkpoints = False # you must use only train.py if you want to load the saved model
load_checkpoints_path = None # or path
load_checkpoints_epoch = None #None

#test_script = 'egs/speech_commands/0.01/word/validation_lmfb_list_word.txt'
test_script = '/n/rd32/mimura/e2e/data/original/feature/eval/script.eval1'
#test_script = '/n/sd3/feng/data/swb/swb/inputs_test.txt'
#test_script = '/n/rd28/mimura/speech_commands/feature/script.e2e.word.testing'

# general config
lmfb_dim = 40
#num_classes = 19146
#num_classes = 10459
num_classes = 47465
eos_id = 1

# network config
frame_stacking = 3 # or False
num_hidden_nodes = 320
num_encoder_layer = 5
encoder_dropout = 0.2
encoder_type = None # 'CNN', 'Wave'
decoder_type = 'Attention' #'CTC' #or 'Attention' 

# training setting
batch_size = 40
max_epoch = 40

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
