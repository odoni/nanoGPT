# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such
# dtype='float32'
# init_from = 'gpt2'
# out_dir = 'out-maquiavel'
# eval_interval = 20 # keep frequent because we'll overfit
# eval_iters = 200
# log_interval = 1 # don't print too too often

# # we expect to overfit on this small dataset, so only save when val improves
# always_save_checkpoint = True

# wandb_log = False # override via command line if you like
# wandb_project = 'maquiavel'
# wandb_run_name = 'gpt2-maquiavel'

# dataset = 'maquiavel'
# batch_size = 64
# block_size = 256 # context of up to 256 previous characters

# # baby GPT model :)
# # n_layer = 6
# # n_head = 6
# # n_embd = 384
# # dropout = 0.2

# learning_rate = 1e-3 # with baby networks can afford to go a bit higher
# max_iters = 20
# lr_decay_iters = 20 # make equal to max_iters usually
# min_lr = 1e-4 # learning_rate / 10 usually
# beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

# warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model

import time

out_dir = 'out-maquiavel'
eval_interval = 5
eval_iters = 40
wandb_log = False # feel free to turn on
wandb_project = 'maquiavel'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'maquiavel'
init_from = 'gpt2-xl' # this is the largest GPT-2 model

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 20

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False
