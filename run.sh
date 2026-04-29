#NCCL_NET=Socket NCCL_DEBUG=WARN DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 RUN_ID=baseline0323_2m ITERATIONS=4000 WARMDOWN_ITERS=700 MUON_MOMENTUM_WARMUP_STEPS=300 MAX_WALLCLOCK_SECONDS=0 VAL_LOSS_EVERY=1000 torchrun --standalone --nproc_per_node=2 train_gpt_mqy.py

#TODO GRAD_CLIP_NORM=0.3 ? 
#
#
# NCCL_NET=Socket NCCL_DEBUG=WARN DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 RUN_ID=baseline torchrun --standalone --nproc_per_node=2 train_gpt.py

# baseline H100x8
# NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
# EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
# ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
# VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
# TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
# TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
# MUON_WD=0.04 ADAM_WD=0.04 \
# MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
# MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
# MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
# ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
# SEED=1337 \
# torchrun --standalone --nproc_per_node=2 train_gpt_mqy.py

# baseline H100x2
# NCCL_NET=Socket NCCL_DEBUG=WARN NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
# EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
# ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
# VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
# TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
# TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
# MUON_WD=0.04 ADAM_WD=0.04 \
# MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
# MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
# MUON_MOMENTUM_WARMUP_STEPS=375 WARMDOWN_ITERS=875 \
# ITERATIONS=2250 MAX_WALLCLOCK_SECONDS=0 EVAL_STRIDE=64 \
# SEED=1337 \
# RUN_ID=baseline0327_2H100 TRAIN_LOG_EVERY=1 VAL_LOSS_EVERY=500 \
# torchrun --standalone --nproc_per_node=2 train_gpt_mqy.py


# baseline+mudd
# NCCL_NET=Socket NCCL_DEBUG=WARN NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
# EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
# ROPE_DIMS=16 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
# VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
# TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
# TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
# MUON_WD=0.04 ADAM_WD=0.04 \
# MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
# MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
# MUON_MOMENTUM_WARMUP_STEPS=375 EVAL_STRIDE=64 \
# ITERATIONS=2250 MAX_WALLCLOCK_SECONDS=0 WARMDOWN_ITERS=875 \
# SEED=1337 \
# RUN_ID=baseline_dc_querywise_LGGG_w256_nsplit4 TRAIN_LOG_EVERY=1 VAL_LOSS_EVERY=500 CHECKPOINT_EVERY=500 USE_MUDD=0 MULTIWAY=0 \
# MUDD_Q_DILATION=2 MUDD_K_DILATION=1 GRAD_CLIP_NORM=0.3 LN_SCALE=1 USE_KV_SHIFT=0 USE_DCMHA=1 WARMUP_STEPS=20 PROFILE=0 \
# torchrun --standalone --nproc_per_node=2 train_gpt_mqy.py


# baseline full train
# NCCL_NET=Socket NCCL_DEBUG=WARN NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
# EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
# ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
# VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
# TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
# TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
# MUON_WD=0.04 ADAM_WD=0.04 \
# MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
# MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
# MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
# ITERATIONS=7182 MAX_WALLCLOCK_SECONDS=0 EVAL_STRIDE=64 \
# SEED=1337 \
# TRAIN_LOG_EVERY=1 VAL_LOSS_EVERY=200 \
# RUN_ID=baseline_full_train \
# torchrun --standalone --nproc_per_node=2 train_gpt_mqy.py


# baseline+dc_LGLL_querywise+kvshift+mudd_querydilation2 full train
# NCCL_NET=Socket NCCL_DEBUG=WARN NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
# EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
# ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
# VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
# TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
# TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
# MUON_WD=0.04 ADAM_WD=0.04 \
# MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
# MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
# MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
# ITERATIONS=7182 MAX_WALLCLOCK_SECONDS=0 EVAL_STRIDE=64 \
# SEED=1337 \
# RUN_ID=baseline_dc_kvshift_mudd_full_train \
# TRAIN_LOG_EVERY=1 VAL_LOSS_EVERY=200 CHECKPOINT_EVERY=0 USE_MUDD=1 MULTIWAY=1 \
# MUDD_Q_DILATION=2 MUDD_K_DILATION=1 GRAD_CLIP_NORM=0.3 LN_SCALE=1 USE_KV_SHIFT=1 USE_DCMHA=1 WARMUP_STEPS=20 PROFILE=0 \
# torchrun --standalone --nproc_per_node=2 train_gpt_mqy.py


# baseline+kvshift+mudd_querydilation2 full train
# NCCL_NET=Socket NCCL_DEBUG=WARN NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
# EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
# ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
# VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
# TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
# TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
# MUON_WD=0.04 ADAM_WD=0.04 \
# MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
# MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
# MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
# ITERATIONS=7182 MAX_WALLCLOCK_SECONDS=0 EVAL_STRIDE=64 \
# SEED=1337 \
# RUN_ID=baseline_full_train_no-x0v0unet \
# TRAIN_LOG_EVERY=1 VAL_LOSS_EVERY=200 CHECKPOINT_EVERY=0 USE_MUDD=0 MULTIWAY=0 \
# MUDD_Q_DILATION=2 MUDD_K_DILATION=1 GRAD_CLIP_NORM=0.3 LN_SCALE=1 USE_KV_SHIFT=0 USE_DCMHA=0 WARMUP_STEPS=20 PROFILE=0 \
# nohup torchrun --standalone --nproc_per_node=2 train_gpt_mqy.py > logs2/${RUN_ID}.log 2>&1 &

# NCCL_NET=Socket NCCL_DEBUG=WARN SEED=42 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
# RUN_ID=baseline0409_debug ITERATIONS=4550 MAX_WALLCLOCK_SECONDS=0 TRAIN_LOG_EVERY=1 VAL_LOSS_EVERY=200 USE_KV_SHIFT=0 USE_DCMHA=0 USE_MUDD=0 MULTIWAY=0 MUDD_Q_DILATION=2 MUDD_K_DILATION=1 \
# torchrun --standalone --nproc_per_node=2 train_gpt0409_base.py


NCCL_NET=Socket NCCL_DEBUG=WARN SEED=42 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
RUN_ID=baseline0409_mudd_q2k1_LnLGGG_wayhalf_recur  MAX_WALLCLOCK_SECONDS=0 TRAIN_LOG_EVERY=1 \
USE_KV_SHIFT=0 USE_DCMHA=0 USE_MUDD=1 MUDD_Q_DILATION=2 MUDD_K_DILATION=1 WARMDOWN_FRAC=0.72 \
TIED_EMBED_LR=0.03 MATRIX_LR=0.022 SCALAR_LR=0.02 \
TRAIN_BATCH_TOKENS=786432 ITERATIONS=4550 VAL_LOSS_EVERY=200 XSA_LAST_N=11 ENABLE_LOOPING_AT=0.35 \
torchrun --standalone --nproc_per_node=2 train_gpt0409.py


# baseline_dc_querywise_LGGG_w256

# default lr 
# TRAIN_BATCH_TOKENS=196608 ITERATIONS=18200 VAL_LOSS_EVERY=800

# TRAIN_BATCH_TOKENS=786432 ITERATIONS=4550 VAL_LOSS_EVERY=200 \ 
# TIED_EMBED_LR=0.03 MATRIX_LR=0.022 SCALAR_LR=0.02 \

# mudd_multiway_q2k1_scale0p1_kvshift

# mudd_multiway_q2k1_ch2_reproduce
# torch._inductor.config.force_disable_caches = True

# full train 

NCCL_NET=Socket NCCL_DEBUG=WARN SEED=42 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
RUN_ID=baseline0409_mudd_q2k2_LnLGGG_wayhalf_recur_full_train_lite \
USE_KV_SHIFT=0 USE_DCMHA=0 USE_MUDD=1 MUDD_Q_DILATION=2 MUDD_K_DILATION=2 \
torchrun --standalone --nproc_per_node=8 train_gpt0409.py


TORCHINDUCTOR_CACHE_DIR=./triton_cache NCCL_NET=Socket NCCL_DEBUG=WARN SEED=423 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
RUN_ID=baseline0409_mudd_qdil2kdil2_2way_vway_base6_vbeforeproj_matched_params_seed423_mha_emb TENSORBOARD_DIR='' WARMUP_STEPS=150 \
USE_KV_SHIFT=0 USE_DCMHA=0 USE_MUDD=1 MUDD_Q_DILATION=2 MUDD_K_DILATION=1 \
TRAIN_BATCH_TOKENS=786432 MAX_WALLCLOCK_SECONDS=600 \
KEEP_UNET=0 NUM_LAYERS=11 MUDD_EMB=1 MLP_MULT=2.78 NUM_KV_HEADS=8 \
torchrun --standalone --nproc_per_node=8 train_gpt0409.py

# sota: val_loss: 2.8037 val_bpb: 1.0854; quantized_ttt val_loss:2.78099004 val_bpb:1.07660791
# TORCHINDUCTOR_CACHE_DIR=./triton_cache NCCL_NET=Socket NCCL_DEBUG=WARN SEED=423 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
# RUN_ID=baseline0409_mudd_qdil2kdil2_2way_vway_base6_vbeforeproj_matched_params_seed423_mha TENSORBOARD_DIR='' WARMUP_STEPS=150 \
# USE_KV_SHIFT=0 USE_DCMHA=0 USE_MUDD=1 MUDD_Q_DILATION=2 MUDD_K_DILATION=1 \
# TRAIN_BATCH_TOKENS=786432 MAX_WALLCLOCK_SECONDS=600 \
# KEEP_UNET=0 NUM_LAYERS=11 MUDD_EMB=0 MLP_MULT=3.5 NUM_KV_HEADS=8 \
# torchrun --standalone --nproc_per_node=8 train_gpt0409.py


# dilated muddformer 
TORCHINDUCTOR_CACHE_DIR=./triton_cache NCCL_NET=Socket NCCL_DEBUG=WARN SEED=423 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
RUN_ID=baseline0409_mudd_seed423  USE_MUDD=1 KEEP_UNET=0 MLP_MULT=3.5 NUM_KV_HEADS=8  WARMUP_STEPS=150 TENSORBOARD_DIR=''  \
torchrun --standalone --nproc_per_node=8 train_gpt0409.py

NCCL_NET=Socket NCCL_DEBUG=WARN SEED=42 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
RUN_ID=baseline0409_660s MAX_WALLCLOCK_SECONDS=660 \
torchrun --standalone --nproc_per_node=8 train_gpt0409_base.py

NCCL_NET=Socket NCCL_DEBUG=WARN SEED=42 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
RUN_ID=baseline0409_720s MAX_WALLCLOCK_SECONDS=720 \
torchrun --standalone --nproc_per_node=8 train_gpt0409_base.py

NCCL_NET=Socket NCCL_DEBUG=WARN SEED=42 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
RUN_ID=baseline0409_540s MAX_WALLCLOCK_SECONDS=540 \
torchrun --standalone --nproc_per_node=8 train_gpt0409_base.py

NCCL_NET=Socket NCCL_DEBUG=WARN SEED=42 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
RUN_ID=baseline0409_480s MAX_WALLCLOCK_SECONDS=480 \
torchrun --standalone --nproc_per_node=8 train_gpt0409_base.py

NCCL_NET=Socket NCCL_DEBUG=WARN SEED=42 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
RUN_ID=baseline0409_420s MAX_WALLCLOCK_SECONDS=420 \
torchrun --standalone --nproc_per_node=8 train_gpt0409_base.py


# configs 
if mode == 'mudd_base':
				self.num_ways = [1]*12 + [2] * looped_num_layers
				self.mudd_q_indices = list(range(0, looped_num_layers, self.mudd_q_dilation))[1:] + [15,16]
				local_window_sizes= [2, None, None,None]*looped_num_layers
			elif mode == 'mudd_base2':
				self.num_ways = [1]*12 + [2] * looped_num_layers
				self.mudd_q_indices = list(range(0, looped_num_layers, self.mudd_q_dilation))[1:] + [15,16]
				local_window_sizes= [None, None, 2,None]*looped_num_layers
			elif mode == 'mudd_base3':
				self.num_ways = [1]*8 + [2] * looped_num_layers
				self.mudd_q_indices = list(range(0, looped_num_layers, self.mudd_q_dilation))[1:] + [15,16]
				local_window_sizes= [None, None, 2,None]*looped_num_layers
			elif mode == 'mudd_base5':
				self.num_ways = [1]*12 + [2] * looped_num_layers
				self.mudd_q_indices = [2,4,6,8,10,13,15,16]#list(range(0, looped_num_layers, self.mudd_q_dilation))[1:] + [15,16]
				local_window_sizes= [None, None, 2,None]*looped_num_layers
			elif mode == 'mudd_base6':
				self.num_ways = [1]*12 + [2] * looped_num_layers
				self.mudd_q_indices = [2,4,6,8,10,12,15,16]
				local_window_sizes= [None, None, 2,None]*looped_num_layers
			elif mode == 'mudd_base7':
				self.num_ways = [1]*12 + [3] * looped_num_layers
				self.mudd_q_indices = [2,4,6,8,10,12,15,16]
				local_window_sizes= [None, None, 2,None]*looped_num_layers
			elif mode == 'mudd_base8':
				self.num_ways = [1]*12 + [3] + [2] * looped_num_layers
				self.mudd_q_indices = [2,4,6,8,10,12,15,16]
				local_window_sizes= [None, None, 2,None]*looped_num_layers
			elif mode == 'mudd_base_lite2':
				self.num_ways = [1]*15 + [2] * looped_num_layers
				self.mudd_q_indices = list(range(0, looped_num_layers, self.mudd_q_dilation))[1:] + [15,16]
				local_window_sizes= [2]*8 + [None, None, 2,None]*looped_num_layers
			elif mode == 'mudd_base_lite3':
				self.num_ways = [1]*12 + [2] * looped_num_layers
				self.mudd_q_indices = list(range(0, looped_num_layers, self.mudd_q_dilation))[1:] + [15,16]
				local_window_sizes= [2]*10 + [None, None, 2,None]*looped_num_layers
			elif mode == 'mudd_base_lite4': 
				self.num_ways = [1]*15 + [2] * looped_num_layers
				self.mudd_q_indices = list(range(0, looped_num_layers, self.mudd_q_dilation))[1:] + [15,16]
				local_window_sizes= [2, None, None,None]*looped_num_layers
			elif mode == 'mudd_base_lite5':
				self.num_ways = [1]*8 + [2] * looped_num_layers
				self.mudd_q_indices = list(range(0, looped_num_layers, self.mudd_q_dilation))[1:] + [15,16]
				local_window_sizes= [2, None, None,None]*looped_num_layers
			elif mode == 'mudd_base_lite6':
				self.num_ways = [1]*12 + [2] * looped_num_layers
				self.mudd_q_indices = [0, 1, 2, 14, 15, 16]
				# self.mudd_q_indices = list(range(0, looped_num_layers, self.mudd_q_dilation))[1:] + [15,16]
				local_window_sizes= [2, None, None,None]*3 + [2, 2, 2, None,None]
			elif mode == 'mudd_vway_only':
				self.num_ways = [1]* looped_num_layers
				self.mudd_q_indices = [15,16]
				local_window_sizes = [None]*looped_num_layers
			elif mode == 'mudd_vway_only15':
				self.num_ways = [1]* looped_num_layers
				self.mudd_q_indices = [15]
				local_window_sizes = [None]*looped_num_layers
			elif mode == 'mudd_vway_only16':
				self.num_ways = [1]* looped_num_layers
				self.mudd_q_indices = [16]
				local_window_sizes = [None]*looped_num_layers
			elif mode == 'mudd_q_all':
				self.mudd_q_indices = list(range(looped_num_layers))
				local_window_sizes = [1] * 8 + [2, 1, None,1] + [2, 1, None,None] + [None, None] #*looped_num_layers # first 8 layers and even layers mix x0 only; 
				self.num_ways = [1]*12 + [2, 1] * looped_num_layers