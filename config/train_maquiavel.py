# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such
dtype='float32'
init_from = 'gpt2-xl'
out_dir = 'out-maquiavel'
eval_interval = 1000 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 1 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True

wandb_log = False # override via command line if you like
wandb_project = 'maquiavel'
wandb_run_name = 'gpt2-xl-maquiavel'

dataset = 'maquiavel'
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
# n_layer = 6
# n_head = 6
# n_embd = 384
# dropout = 0.2

learning_rate = 1e-1 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
# min_lr = 1e-4 # learning_rate / 10 usually
# beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

# warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
