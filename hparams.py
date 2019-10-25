## CONFIG

#train_script = '/n/sd3/feng/data/swb/swb/inputs_train.txt'
#train_script = '/n/rd23/ueno/e2e/data/aps/script.word.sort_xlen.shorter_than_12s'
train_script = '/n/rd32/mimura/e2e/data/original/script/aps_sps/bpe/script.bpe.sort_xlen'
train_script = '/n/rd32/mimura/e2e/data/original/script/aps_sps/bpe/script.bpe.sort_xlen'
#train_script = 'test'#'/n/rd23/ueno/e2e/data/aps/script.word.sort_xlen.shorter_than_12s'
#train_script = '/n/rd28/mimura/speech_commands/feature/script.e2e.word.training.shuffled'
#train_script = 'egs/speech_commands/0.02/word/training_lmfb_list_word_id_random.txt'
save_dir = 'checkpoints.bpe.specaug.multi'

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
#num_classes = 13245
#num_classes = 24826
#num_classes = 34331
num_classes = 10459
eos_id = 1

# network config
frame_stacking = 3 # or False
num_hidden_nodes = 320
num_encoder_layer = 5
encoder_dropout = 0.2
encoder_type = 'None' # 'CNN', 'Wave'
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
