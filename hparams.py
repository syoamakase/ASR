## CONFIG

train_script = '/n/rd23/ueno/e2e/data/aps/script.word.sort_xlen.shorter_than_12s'
save_dir = 'checkpoints'

load_checkpoints = True # you must use only train.py if you want to load the saved model
load_checkpoints_path = '/n/rd26/ueno/e2e/trial_exp/Class_Balanced_Loss/model.word.mb40_LS_APS_ClassBlancedLoss_beta0.9'#None # or path
load_checkpoints_epoch = None #None

test_script = '/n/rd32/mimura/e2e/data/original/feature/eval/script.eval1'

# general config
lmfb_dim = 40
num_classes = 19146
eos_id = 1

# network config
frame_stacking = True
num_hidden_nodes = 320
num_encoder_layer = 5
frame_stacking = 3
encoder_dropout = 0.2

# trainig setting
batch_size = 40
max_epoch = 40

# inference config
max_decoder_seq_len = 200
beam_width = 4
shallow_fusion = False
if shallow_fusion:
    LM_path = None

score_func = 'log_softmax' # or 'logit' or 'softmax'

# for previous version
legacy = True 
