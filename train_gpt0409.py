import collections,copy,glob,io,lzma,math,os
from pathlib import Path
import random,re,subprocess,sys,time,uuid,numpy as np,sentencepiece as spm,torch,torch.distributed as dist,torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import Tensor,nn
from torch.utils.tensorboard import SummaryWriter
from flash_attn_interface import flash_attn_func as flash_attn_3_func
from einops import rearrange
from typing import Optional, Tuple
class Hyperparameters:data_dir=os.environ.get('DATA_DIR','./data/');seed=int(os.environ.get('SEED',1337));run_id=os.environ.get('RUN_ID',str(uuid.uuid4()));iterations=int(os.environ.get('ITERATIONS',20000));warmdown_frac=float(os.environ.get('WARMDOWN_FRAC',.72));warmup_steps=int(os.environ.get('WARMUP_STEPS',20));train_batch_tokens=int(os.environ.get('TRAIN_BATCH_TOKENS',786432));train_seq_len=int(os.environ.get('TRAIN_SEQ_LEN',2048));train_log_every=int(os.environ.get('TRAIN_LOG_EVERY',500));max_wallclock_seconds=float(os.environ.get('MAX_WALLCLOCK_SECONDS',6e2));val_batch_tokens=int(os.environ.get('VAL_BATCH_TOKENS',524288));eval_seq_len=int(os.environ.get('EVAL_SEQ_LEN',2048));val_loss_every=int(os.environ.get('VAL_LOSS_EVERY',4000));sliding_window_enabled=bool(int(os.environ.get('SLIDING_WINDOW_ENABLED','1')));vocab_size=int(os.environ.get('VOCAB_SIZE',8192));num_layers=int(os.environ.get('NUM_LAYERS',11));xsa_last_n=int(os.environ.get('XSA_LAST_N',11));model_dim=int(os.environ.get('MODEL_DIM',512));embedding_dim=int(os.environ.get('EMBEDDING_DIM',512));num_kv_heads=int(os.environ.get('NUM_KV_HEADS',4));num_heads=int(os.environ.get('NUM_HEADS',8));mlp_mult=float(os.environ.get('MLP_MULT',4.));skip_gates_enabled=bool(int(os.environ.get('SKIP_GATES_ENABLED','1')));tie_embeddings=bool(int(os.environ.get('TIE_EMBEDDINGS','1')));logit_softcap=float(os.environ.get('LOGIT_SOFTCAP',3e1));rope_base=float(os.environ.get('ROPE_BASE',1e4));rope_dims=int(os.environ.get('ROPE_DIMS',16));rope_train_seq_len=int(os.environ.get('ROPE_TRAIN_SEQ_LEN',2048));ln_scale=bool(int(os.environ.get('LN_SCALE','1')));qk_gain_init=float(os.environ.get('QK_GAIN_INIT',5.));num_loops=int(os.environ.get('NUM_LOOPS',2));loop_start=int(os.environ.get('LOOP_START',3));loop_end=int(os.environ.get('LOOP_END',5));enable_looping_at=float(os.environ.get('ENABLE_LOOPING_AT',.35));parallel_residual_start=int(os.environ.get('PARALLEL_RESIDUAL_START',7));min_lr=float(os.environ.get('MIN_LR',.0));embed_lr=float(os.environ.get('EMBED_LR',.6));head_lr=float(os.environ.get('HEAD_LR',.008));tied_embed_lr=float(os.environ.get('TIED_EMBED_LR',.03));tied_embed_init_std=float(os.environ.get('TIED_EMBED_INIT_STD',.005));matrix_lr=float(os.environ.get('MATRIX_LR',.022));scalar_lr=float(os.environ.get('SCALAR_LR',.02));muon_momentum=float(os.environ.get('MUON_MOMENTUM',.99));muon_backend_steps=int(os.environ.get('MUON_BACKEND_STEPS',5));muon_momentum_warmup_start=float(os.environ.get('MUON_MOMENTUM_WARMUP_START',.92));muon_momentum_warmup_steps=int(os.environ.get('MUON_MOMENTUM_WARMUP_STEPS',1500));muon_row_normalize=bool(int(os.environ.get('MUON_ROW_NORMALIZE','1')));beta1=float(os.environ.get('BETA1',.9));beta2=float(os.environ.get('BETA2',.95));adam_eps=float(os.environ.get('ADAM_EPS',1e-08));grad_clip_norm=float(os.environ.get('GRAD_CLIP_NORM',.3));eval_stride=int(os.environ.get('EVAL_STRIDE',64));muon_beta2=float(os.environ.get('MUON_BETA2',.95));adam_wd=float(os.environ.get('ADAM_WD',.02));muon_wd=float(os.environ.get('MUON_WD',.095));embed_wd=float(os.environ.get('EMBED_WD',.085));ema_decay=float(os.environ.get('EMA_DECAY',.9965));ttt_enabled=bool(int(os.environ.get('TTT_ENABLED','0')));ttt_lr=float(os.environ.get('TTT_LR',.005));ttt_epochs=int(os.environ.get('TTT_EPOCHS',3));ttt_momentum=float(os.environ.get('TTT_MOMENTUM',.9));ttt_chunk_tokens=int(os.environ.get('TTT_CHUNK_TOKENS',32768));etlb_enabled=bool(int(os.environ.get('ETLB_ENABLED','0')));etlb_lr=float(os.environ.get('ETLB_LR',.05));etlb_steps=int(os.environ.get('ETLB_STEPS',5));etlb_clip=float(os.environ.get('ETLB_CLIP',3.));compressor=os.environ.get('COMPRESSOR','brotli');gptq_calibration_batches=int(os.environ.get('GPTQ_CALIBRATION_BATCHES',64));gptq_reserve_seconds=float(os.environ.get('GPTQ_RESERVE_SECONDS',12.));matrix_bits=int(os.environ.get('MATRIX_BITS',6));embed_bits=int(os.environ.get('EMBED_BITS',8));matrix_clip_sigmas=float(os.environ.get('MATRIX_CLIP_SIGMAS',12.85));embed_clip_sigmas=float(os.environ.get('EMBED_CLIP_SIGMAS',2e1));distributed='RANK'in os.environ and'WORLD_SIZE'in os.environ;rank=int(os.environ.get('RANK','0'));world_size=int(os.environ.get('WORLD_SIZE','1'));local_rank=int(os.environ.get('LOCAL_RANK','0'));is_main_process=rank==0;grad_accum_steps=8//world_size;datasets_dir=os.path.join(data_dir,'datasets',f"fineweb10B_sp{vocab_size}");train_files=os.path.join(datasets_dir,'fineweb_train_*.bin');val_files=os.path.join(datasets_dir,'fineweb_val_*.bin');tokenizer_path=os.path.join(data_dir,'tokenizers',f"fineweb_{vocab_size}_bpe.model");logfile=f"logs/{run_id}.txt";model_path='final_model.pt';quantized_model_path='final_model.int6.ptz';tensorboard_dir = os.environ.get("TENSORBOARD_DIR", "./logs/tensorboard");use_mudd=bool(int(os.environ.get('USE_MUDD','0')));mudd_q_dilation=int(os.environ.get('MUDD_Q_DILATION','1'));mudd_k_dilation=int(os.environ.get('MUDD_K_DILATION','1'));mudd_prenorm=bool(int(os.environ.get('MUDD_PRENORM','0')));use_kv_shift=bool(int(os.environ.get('USE_KV_SHIFT','0')));use_dcmha=bool(int(os.environ.get('USE_DCMHA','0')))
_logger_hparams=None
def set_logging_hparams(h):global _logger_hparams;_logger_hparams=h
def log(msg,console=True):
	if _logger_hparams is None:print(msg);return
	if _logger_hparams.is_main_process:
		if console:print(msg)
		if _logger_hparams.logfile is not None:
			with open(_logger_hparams.logfile,'a',encoding='utf-8')as f:print(msg,file=f)
class ValidationData:
	def __init__(self,h,device):
		self.sp=spm.SentencePieceProcessor(model_file=h.tokenizer_path)
		if int(self.sp.vocab_size())!=h.vocab_size:raise ValueError(f"VOCAB_SIZE={h.vocab_size} does not match tokenizer vocab_size={int(self.sp.vocab_size())}")
		self.val_tokens=load_validation_tokens(h.val_files,h.eval_seq_len);self.base_bytes_lut,self.has_leading_space_lut,self.is_boundary_token_lut=build_sentencepiece_luts(self.sp,h.vocab_size,device)
def build_sentencepiece_luts(sp,vocab_size,device):
	sp_vocab_size=int(sp.vocab_size());assert sp.piece_to_id('▁')!=sp.unk_id(),"Tokenizer must have '▁' (space) as its own token for correct BPB byte counting";table_size=max(sp_vocab_size,vocab_size);base_bytes_np=np.zeros((table_size,),dtype=np.int16);has_leading_space_np=np.zeros((table_size,),dtype=np.bool_);is_boundary_token_np=np.ones((table_size,),dtype=np.bool_)
	for token_id in range(sp_vocab_size):
		if sp.is_control(token_id)or sp.is_unknown(token_id)or sp.is_unused(token_id):continue
		is_boundary_token_np[token_id]=False
		if sp.is_byte(token_id):base_bytes_np[token_id]=1;continue
		piece=sp.id_to_piece(token_id)
		if piece.startswith('▁'):has_leading_space_np[token_id]=True;piece=piece[1:]
		base_bytes_np[token_id]=len(piece.encode('utf-8'))
	return torch.tensor(base_bytes_np,dtype=torch.int16,device=device),torch.tensor(has_leading_space_np,dtype=torch.bool,device=device),torch.tensor(is_boundary_token_np,dtype=torch.bool,device=device)
def load_validation_tokens(pattern,seq_len):
	files=[Path(p)for p in sorted(glob.glob(pattern))]
	if not files:raise FileNotFoundError(f"No files found for pattern: {pattern}")
	tokens=torch.cat([load_data_shard(file)for file in files]).contiguous();usable=(tokens.numel()-1)//seq_len*seq_len
	if usable<=0:raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
	return tokens[:usable+1]
def load_data_shard(file):
	header_bytes=256*np.dtype('<i4').itemsize;token_bytes=np.dtype('<u2').itemsize;header=np.fromfile(file,dtype='<i4',count=256)
	if header.size!=256 or int(header[0])!=20240520 or int(header[1])!=1:raise ValueError(f"Unexpected shard header for {file}")
	num_tokens=int(header[2]);expected_size=header_bytes+num_tokens*token_bytes
	if file.stat().st_size!=expected_size:raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
	tokens_np=np.fromfile(file,dtype='<u2',count=num_tokens,offset=header_bytes)
	if tokens_np.size!=num_tokens:raise ValueError(f"Short read for {file}")
	return torch.from_numpy(tokens_np.astype(np.uint16,copy=False))
_SHARD_HEADER_BYTES=256*np.dtype('<i4').itemsize
_SHARD_NTOKENS_CACHE={}
_MMAP_CACHE={}
def _read_num_tokens(file):
	key=str(file);cached=_SHARD_NTOKENS_CACHE.get(key)
	if cached is not None:return cached
	header=np.fromfile(file,dtype='<i4',count=256)
	if header.size!=256 or int(header[0])!=20240520 or int(header[1])!=1:raise ValueError(f"Unexpected shard header for {file}")
	n=int(header[2]);_SHARD_NTOKENS_CACHE[key]=n;return n
def _get_shard_memmap(file):
	key=str(file);mm=_MMAP_CACHE.get(key)
	if mm is not None:return mm
	n=_read_num_tokens(file);mm=np.memmap(file,mode='r',dtype='<u2',offset=_SHARD_HEADER_BYTES,shape=(n,));_MMAP_CACHE[key]=mm;return mm
class ShuffledSequenceLoader:
	def __init__(self,h,device):
		self.world_size=h.world_size;self.seq_len=h.train_seq_len;self.device=device;all_files=[Path(p)for p in sorted(glob.glob(h.train_files))]
		if not all_files:raise FileNotFoundError(f"No files found for pattern: {h.train_files}")
		self.files=all_files[h.rank::h.world_size];self.rng=np.random.Generator(np.random.PCG64(h.rank));self.num_tokens=[_read_num_tokens(f)for f in self.files];self.start_inds=[[]for _ in self.files]
		for si in range(len(self.files)):self._reset_shard(si)
	def _reset_shard(self,si):max_phase=min(self.seq_len-1,max(0,self.num_tokens[si]-self.seq_len-1));phase=int(self.rng.integers(max_phase+1))if max_phase>0 else 0;num_sequences=(self.num_tokens[si]-1-phase)//self.seq_len;sequence_order=self.rng.permutation(num_sequences);self.start_inds[si]=(phase+sequence_order*self.seq_len).tolist()
	def next_batch(self,global_tokens,grad_accum_steps):
		device_tokens=global_tokens//(self.world_size*grad_accum_steps);device_batch_size=device_tokens//self.seq_len;remaining=np.array([len(s)for s in self.start_inds],dtype=np.float64);x=torch.empty((device_batch_size,self.seq_len),dtype=torch.int64);y=torch.empty((device_batch_size,self.seq_len),dtype=torch.int64)
		for bi in range(device_batch_size):
			total=remaining.sum()
			if total<=0:
				for si in range(len(self.files)):self._reset_shard(si)
				remaining=np.array([len(s)for s in self.start_inds],dtype=np.float64);total=remaining.sum()
			probs=remaining/total;si=int(self.rng.choice(len(self.files),p=probs));start_ind=self.start_inds[si].pop();remaining[si]-=1;mm=_get_shard_memmap(self.files[si]);window=torch.as_tensor(np.array(mm[start_ind:start_ind+self.seq_len+1],dtype=np.int64));x[bi]=window[:-1];y[bi]=window[1:]
		return x.to(self.device,non_blocking=True),y.to(self.device,non_blocking=True)
class RMSNorm(nn.Module):
	def __init__(self,eps=None):super().__init__();self.eps=eps
	def forward(self,x):return F.rms_norm(x,(x.size(-1),),eps=self.eps)
class CastedLinear(nn.Linear):
	def forward(self,x):w=self.weight.to(x.dtype);bias=self.bias.to(x.dtype)if self.bias is not None else None;return F.linear(x,w,bias)
class Rotary(nn.Module):
	def __init__(self,dim,base=1e4,train_seq_len=1024,rope_dims=0):super().__init__();self.dim=dim;self.base=base;self.train_seq_len=train_seq_len;self.rope_dims=rope_dims if rope_dims>0 else dim;inv_freq=1./base**(torch.arange(0,self.rope_dims,2,dtype=torch.float32)/self.rope_dims);self.register_buffer('inv_freq',inv_freq,persistent=False);self._seq_len_cached=0;self._cos_cached=None;self._sin_cached=None
	def forward(self,seq_len,device,dtype):
		if self._cos_cached is None or self._sin_cached is None or self._seq_len_cached!=seq_len or self._cos_cached.device!=device:
			rd=self.rope_dims
			if seq_len>self.train_seq_len:scale=seq_len/self.train_seq_len;new_base=self.base*scale**(rd/(rd-2));inv_freq=1./new_base**(torch.arange(0,rd,2,dtype=torch.float32,device=device)/rd)
			else:inv_freq=self.inv_freq.to(device)
			t=torch.arange(seq_len,device=device,dtype=inv_freq.dtype);freqs=torch.outer(t,inv_freq);self._cos_cached=freqs.cos()[None,:,None,:];self._sin_cached=freqs.sin()[None,:,None,:];self._seq_len_cached=seq_len
		return self._cos_cached.to(dtype=dtype),self._sin_cached.to(dtype=dtype)
def apply_rotary_emb(x,cos,sin,rope_dims=0):
	if rope_dims>0 and rope_dims<x.size(-1):x_rope,x_pass=x[...,:rope_dims],x[...,rope_dims:];half=rope_dims//2;x1,x2=x_rope[...,:half],x_rope[...,half:];x_rope=torch.cat((x1*cos+x2*sin,x1*-sin+x2*cos),dim=-1);return torch.cat((x_rope,x_pass),dim=-1)
	half=x.size(-1)//2;x1,x2=x[...,:half],x[...,half:];return torch.cat((x1*cos+x2*sin,x1*-sin+x2*cos),dim=-1)
def unbind(ary, n, dim=0):
	return [torch.squeeze(a, dim=dim) for a in torch.split(ary, ary.shape[dim] // n, dim=dim)]
def _atten_context(query, key, value, atten_mask, pre_proj_dw_args, post_proj_dw_args, dtype=torch.bfloat16):
	logits = query @ key.transpose(-2, -1)
	if pre_proj_dw_args is not None: logits = _cross_head_proj(logits, *pre_proj_dw_args)
	logits = torch.where(atten_mask, logits, torch.finfo(dtype).min)
	logits = logits.to(torch.float32)
	probs = logits.softmax(-1)
	probs = probs.to(torch.bfloat16)
	if post_proj_dw_args is not None: probs = _cross_head_proj(probs, *post_proj_dw_args)
	probs = torch.where(atten_mask, probs, 0)
	o = probs @ value
	return o
def _cross_head_proj(inputs, sw, qw1, qw2, kw1, kw2, qdd, kdd, loop_over_dynamic_hd=True):
	out = inputs + torch.einsum('BNTS,NM->BMTS', inputs, sw) if sw is not None else inputs
	if loop_over_dynamic_hd:
		for i in range(2):
			if qw1 is not None:
				qhidden = (inputs * qw1[..., i, :].transpose(-2, -1).unsqueeze(-1)).sum(1)
				qout = qhidden.unsqueeze(1) * qw2[..., i, :].transpose(-2, -1).unsqueeze(-1)
				out = out + qout
			if kw1 is not None:
				khidden = (inputs * kw1[..., i, :].transpose(-2, -1).unsqueeze(-2)).sum(1)
				kout = khidden.unsqueeze(1) * kw2[..., i, :].transpose(-2, -1).unsqueeze(-2)
				out = out + kout
		if qdd is not None:
			qdout = inputs * qdd.transpose(-2, -1).unsqueeze(-1); out = out + qdout
		if kdd is not None:
			kdout = inputs * kdd.transpose(-2, -1).unsqueeze(-2); out = out + kdout
	else:
		x_inter = torch.einsum('BNTS, BTIN->BTSI', inputs, qw1)
		qw_out = torch.einsum('BTSI, BTIN->BNTS', x_inter, qw2)
		out = out + qw_out
		out = out + torch.einsum('BNTS, BTN->BNTS', inputs, qdd)
	return out
def slice_dw(sw, qw1, qw2, kw1, kw2, qdd, kdd, start, stop, kv_start):
	return (sw,
		qw1[:, start : stop] if qw1 is not None else None,
		qw2[:, start : stop] if qw2 is not None else None,
		kw1[:, kv_start : stop] if kw1 is not None else None,
		kw2[:, kv_start : stop] if kw2 is not None else None,
		qdd[:, start : stop] if qdd is not None else None,
		kdd[:, kv_start : stop] if kdd is not None else None)
def make_window_mask(t, window_size):
	col_idx = torch.tile(torch.arange(t).unsqueeze(0), [t, 1])
	row_idx = torch.tile(torch.arange(t).unsqueeze(1), [1, t])
	bias_mask = (col_idx + window_size >= row_idx).tril().view(t, t)
	return bias_mask[None, None, :, :]
class CrossHeadProjection(nn.Module):
	def __init__(self, mode, num_heads=16, num_groups=1, dtype=torch.bfloat16, use_sw=False):
		super().__init__()
		self.mode = mode;self.use_sw = use_sw;self.num_heads = num_heads;self.num_groups = num_groups;self.num_heads_per_group = self.num_heads // self.num_groups
		if self.use_sw:self.w = nn.parameter.Parameter(data=torch.zeros(self.num_groups, self.num_heads_per_group, self.num_heads_per_group, dtype=dtype))
		else:self.register_buffer('w', torch.eye(self.num_heads_per_group, dtype=dtype).expand(self.num_groups, self.num_heads_per_group, self.num_heads_per_group))
	def forward(self, inputs, dws:Optional[Tuple[Tensor,Tensor, Tensor,Tensor, Tensor,Tensor]]=None, query_vec=None, key_vec=None, proj_w:Optional[Tensor]=None, fast_infer=True):
		if proj_w is not None:
			ret = torch.einsum('BNTS,BSNM->BMTS', inputs, proj_w)
		else:
			assert dws is not None
			qw1, qw2, kw1, kw2, qdd, kdd = dws
			inputs = inputs.unsqueeze(1)
			ret = torch.einsum('BGMTS,GMN->BGNTS', inputs, self.w) if self.use_sw else inputs
			if fast_infer:
				inputs_label = 'BGMTS';hidden_sym = 'I'; hidden_label = inputs_label.replace('M', 'I')
				for sym, (w1, w2) in zip(['T', 'S'], [(qw1, qw2), (kw1, kw2)]):
					dw_label = f'B{sym}G{hidden_sym}M';dynamic_hidden_dim = w1.shape[dw_label.index(hidden_sym)]
					eqn1 = f'{inputs_label},{dw_label}->{hidden_label}';eqn2 = f'{hidden_label},{dw_label}->{inputs_label}'
					for i in range(dynamic_hidden_dim):
						hidden = torch.einsum(eqn1.replace(hidden_sym, ''), inputs, w1[..., i, :])
						out = torch.einsum(eqn2.replace(hidden_sym, ''), hidden, w2[..., i, :])
						ret = ret + out
				del out
				for sym, dd in zip(['T', 'S'], [qdd, kdd]):
					dd_label = f'B{sym}GM';dout = torch.einsum(f'{inputs_label},{dd_label}->{inputs_label}', inputs, dd)
					ret = ret + dout
				del dout
			else:
				x_inter = torch.einsum('BGNTS, BTGIN->BGTSI', inputs, qw1);qw_out = torch.einsum('BGTSI, BTGIN->BGNTS', x_inter, qw2);ret = ret + qw_out
				x_inter = torch.einsum('BGNTS, BSGIN->BGTSI', inputs, kw1);kw_out = torch.einsum('BGTSI, BSGIN->BGNTS', x_inter, kw2);ret = ret + kw_out
				ret = ret + torch.einsum('BGNTS, BTGN->BGNTS', inputs, qdd);ret = ret + torch.einsum('BGNTS, BSGN->BGNTS', inputs, kdd)
			ret = ret.squeeze(1)
		return ret
class DynamicWeightProjection(nn.Module):
	def __init__(self, dim, num_heads=32, num_groups=1, residual=True, query_input_dim=4096, dynamic_squeeze_ratio=16, dynamic_w_hidden_dim=128,n_splits=4, dtype=torch.bfloat16,use_sw=False):
		super().__init__()
		self.num_heads = num_heads;self.num_groups = num_groups;self.query_input_dim = query_input_dim;self.dynamic_squeeze_ratio = dynamic_squeeze_ratio;self.dynamic_w_hidden_dim = dynamic_w_hidden_dim
		self.dw_hidden_activation = nn.GELU();self.num_heads_per_group = self.num_heads // self.num_groups;self.dw_activation = nn.Tanh();self.dw1_norm = RMSNorm();self.use_sw = use_sw
		self.pre_proj = CrossHeadProjection('pre', num_heads=self.num_heads, use_sw=use_sw, dtype=dtype);self.post_proj = CrossHeadProjection('post', num_heads=self.num_heads, use_sw=use_sw, dtype=dtype)
		self.n_splits = n_splits;dynamic_hidden_dim = self.num_heads_per_group // self.dynamic_squeeze_ratio;self.dynamic_hidden_dim = dynamic_hidden_dim
		rank = 2;qkw_std = 0.02 / (math.sqrt(2*self.num_heads*rank) * (self.num_heads + rank));dd_std = 0.05 * math.sqrt(2/(self.num_heads + dim));dw1_std = math.sqrt(2/(dim+self.dynamic_w_hidden_dim));C = n_splits
		if n_splits == 4:
			self.dw1 = nn.parameter.Parameter(torch.zeros(self.query_input_dim, self.num_groups, C, self.dynamic_w_hidden_dim, dtype=dtype).normal_(mean=0,std=dw1_std))
			G, K, M = self.num_groups, self.dynamic_w_hidden_dim, self.num_heads_per_group;I = dynamic_hidden_dim * 2
			self.qkw = nn.parameter.Parameter(torch.zeros([G, C, K, I, M], dtype=dtype).normal_(mean=0,std=qkw_std))
			self.dd = nn.parameter.Parameter(torch.zeros(self.query_input_dim, self.num_groups, self.num_heads_per_group * C, dtype=dtype).normal_(mean=0,std=dd_std))
		elif n_splits == 2:
			self.dw1 = nn.parameter.Parameter(torch.zeros((self.num_groups * C * rank * self.dynamic_w_hidden_dim), self.query_input_dim, dtype=dtype).normal_(mean=0,std=dw1_std))
			G, K, M = self.num_groups, self.dynamic_w_hidden_dim, self.num_heads_per_group;I = dynamic_hidden_dim * 2
			self.qkw = nn.parameter.Parameter(torch.zeros((G* C* rank*K), (G*C*rank*K), dtype=dtype).normal_(mean=0,std=qkw_std))
			self.dd = nn.parameter.Parameter(torch.zeros(self.num_groups * self.num_heads_per_group * C, self.query_input_dim, dtype=dtype).normal_(mean=0,std=dd_std))
		self.norm_scale = nn.Parameter(torch.ones(self.num_groups, dtype=dtype) * 0.001);self.merge_weights()
	def merge_weights(self):
		if self.use_sw:self.sw = nn.parameter.Parameter(torch.stack([self.pre_proj.w, self.post_proj.w]).squeeze(1) + torch.eye(self.num_heads) ).to(self.dw1.device)
		else:self.sw = (torch.eye(self.num_heads).expand(2,self.num_heads,self.num_heads)).to(self.dw1.device)
	def forward(self,query_vec,KW:Optional[torch.Tensor]=None, gen_cache:Optional[bool]=False, keep_group_dim=False):
		if self.n_splits == 4:
			dw_hidden = torch.einsum('BTD,DGCK->BTGCK', query_vec, self.dw1);dw_hidden = self.dw_hidden_activation(dw_hidden)
			w1, w2 = torch.split(torch.einsum('BTGCK,GCKIM->BTGCIM', dw_hidden, self.qkw), self.qkw.shape[-2]//2, dim=-2)
			w1 = self.dw1_norm(w1);w2 = self.dw1_norm(w2) * self.norm_scale
			if not keep_group_dim:w1 = w1.squeeze(2);w2 = w2.squeeze(2)
			pre_qw1, pre_kw1, post_qw1, post_kw1 = unbind(w1, 4, dim=-3);pre_qw2, pre_kw2, post_qw2, post_kw2 = unbind(w2, 4, dim=-3)
			dd = torch.einsum('BTD,DGM->BTGM', query_vec, self.dd);dd = self.dw_activation(dd)
			if not keep_group_dim:dd = dd.squeeze(-2)
			pre_qdd, pre_kdd, post_qdd, post_kdd = torch.split(dd, dd.shape[-1] // 4, dim=-1)
			key_wise = False
			if not key_wise:pre_kdd, post_kdd = None, None;pre_kw1, post_kw1, pre_kw2, post_kw2 = None, None, None, None
		elif self.n_splits == 2:
			dw_hidden = F.linear(query_vec, self.dw1.to(query_vec.dtype));dw_hidden = self.dw_hidden_activation(dw_hidden)
			dw = F.linear(dw_hidden, self.qkw.to(dw_hidden.dtype));dw = rearrange(dw, 'B T (C R M) -> B T C R M', R=2, M=self.num_heads_per_group)
			dw = self.dw1_norm(dw);dw1, dw2 = torch.split(dw, dw.shape[-3]//2, dim=-3);dw2 = dw2 * self.norm_scale
			pre_qw1, post_qw1 = unbind(dw1, 2, dim=-3);pre_qw2, post_qw2 = unbind(dw2, 2, dim=-3)
			pre_kw1, pre_kw2 = None, None;post_kw1, post_kw2 = None, None
			dd = F.linear(query_vec, self.dd.to(query_vec.dtype));dd = self.dw_activation(dd)
			pre_qdd, post_qdd = torch.split(dd, dd.shape[-1] // 2, dim=-1);pre_kdd, post_kdd = None, None
		pre_dw_args = (pre_qw1, pre_qw2, pre_kw1, pre_kw2, pre_qdd, pre_kdd);post_dw_args = (post_qw1, post_qw2, post_kw1, post_kw2, post_qdd, post_kdd)
		if gen_cache:
			pre_kw = torch.einsum('BSGIM, BSGIN->BSMN', pre_kw1, pre_kw2) + torch.diag_embed(pre_kdd.squeeze(2))
			post_kw = torch.einsum('BSGIM, BSGIN->BSMN', post_kw1, post_kw2) + torch.diag_embed(post_kdd.squeeze(2))
			KW = torch.stack((pre_kw, post_kw), dim=-3)
		return pre_dw_args, post_dw_args, KW
class MultiwayDynamicDenseBlock(nn.Module):
	def __init__(self, dim, lidx, last_layer=False, multiway=False, q_dilation=1, k_dilation=1, base_layer=1, mudd_in_channels=None, num_ways=4, local_window_size=None):
		super().__init__()
		self.norm = RMSNorm()
		if multiway:self.C = num_ways if not last_layer else 1
		else:self.C = 1
		self.lidx = lidx
		if local_window_size is None:l = base_layer + (lidx + 1) // k_dilation
		else:l = local_window_size
		self.local_window_size = local_window_size;hid_dim, out_dim = l * self.C, l * self.C;self.dim = dim;self.mudd_in_channels = mudd_in_channels or dim
		self.w1 = CastedLinear(self.mudd_in_channels, hid_dim, bias=False);self.act = nn.GELU();self.w2 = CastedLinear(hid_dim, out_dim, bias=True);self.w2._zero_init = True;self.w2.bias.data.fill_(0.0)
		self.channels = dim // 1;self.scale = nn.Parameter(torch.ones(self.channels * self.C, dtype=torch.float32)*0.1)
	def forward(self, x):
		x = self.norm(x[:,:,:self.mudd_in_channels])
		dw = self.w2(self.act(self.w1(x)))
		dw = rearrange(dw, 'B T (C L) -> C B T L', C=self.C)
		return dw
	def layer_mix(self, x, all_hids, dw):
		L = dw.shape[3];hids = all_hids[-L:];channels = self.channels;scale = self.scale.to(dtype=hids[0].dtype).view(self.C, 1, 1, -1)
		norm = lambda x:x
		weighted = dw[:, :, :, 0, None] * norm(hids[0][:, :, :channels])
		for j in range(1, L):weighted = weighted + dw[:, :, :, j, None] * norm(hids[j][:, :, :channels])
		normed = F.rms_norm(weighted, (weighted.size(-1),)) * scale
		result = x + F.pad(normed, (0, hids[-1].size(-1) - channels))
		return tuple(result[c] for c in range(self.C)) if self.C > 1 else result[0]
class CausalSelfAttention(nn.Module):
	def __init__(self,dim,num_heads,num_kv_heads,rope_base,qk_gain_init,train_seq_len,use_kv_shift=False,attn_window_size=None,use_dcmha=False):
		super().__init__()
		if dim%num_heads!=0:raise ValueError('model_dim must be divisible by num_heads')
		if num_heads%num_kv_heads!=0:raise ValueError('num_heads must be divisible by num_kv_heads')
		self.num_heads=num_heads;self.num_kv_heads=num_kv_heads;self.head_dim=dim//num_heads;self.attn_window_size=attn_window_size;self.use_dcmha=use_dcmha;self._cached_window_mask=None
		if self.use_dcmha:
			n_splits=4;self.dyn_w_proj=DynamicWeightProjection(dim,num_heads=self.num_heads,query_input_dim=dim,dynamic_squeeze_ratio=self.num_heads//2,dynamic_w_hidden_dim=self.num_heads*n_splits,n_splits=n_splits,dtype=torch.bfloat16,use_sw=False)
		if self.head_dim%2!=0:raise ValueError('head_dim must be even for RoPE')
		kv_dim=self.num_kv_heads*self.head_dim;self.c_q=CastedLinear(dim,dim,bias=False);self.c_k=CastedLinear(dim,kv_dim,bias=False);self.c_v=CastedLinear(dim,kv_dim,bias=False);self.proj=CastedLinear(dim,dim,bias=False);self.proj._zero_init=True;self.q_gain=nn.Parameter(torch.full((num_heads,),qk_gain_init,dtype=torch.float32));self.rope_dims=0;self.rotary=Rotary(self.head_dim,base=rope_base,train_seq_len=train_seq_len);self.use_xsa=False
		self.use_kv_shift=use_kv_shift
		if use_kv_shift:self.kv_shift=CastedLinear(dim,num_kv_heads*2,bias=False);nn.init.zeros_(self.kv_shift.weight)
	def _xsa_efficient(self,y,v):B,T,H,D=y.shape;Hkv=v.size(-2);group=H//Hkv;y_g=y.reshape(B,T,Hkv,group,D);vn=F.normalize(v,dim=-1).unsqueeze(-2);proj=(y_g*vn).sum(dim=-1,keepdim=True)*vn;return(y_g-proj).reshape(B,T,H,D)
	def forward(self,x):
		if isinstance(x, tuple):
			if len(x) == 3:
				xq, xk, xv = x
			elif len(x) == 2: # qk share one way
				xq, xv = x
				xk = xq
			bsz, seqlen, dim = xq.shape
		else:
			bsz, seqlen, dim = x.shape
			xq, xk, xv = x, x, x
		q=self.c_q(xq).reshape(bsz,seqlen,self.num_heads,self.head_dim);k=self.c_k(xk).reshape(bsz,seqlen,self.num_kv_heads,self.head_dim);v=self.c_v(xv).reshape(bsz,seqlen,self.num_kv_heads,self.head_dim)
		if self.use_kv_shift:
			kg,vg=torch.sigmoid(self.kv_shift(xv)).reshape(bsz,seqlen,self.num_kv_heads,2).chunk(2,dim=-1)
			k_prev=torch.cat([k[:,:1],k[:,:-1]],dim=1);v_prev=torch.cat([v[:,:1],v[:,:-1]],dim=1)
			k=k*kg+(1-kg)*k_prev;v=v*vg+(1-vg)*v_prev
		q=F.rms_norm(q,(q.size(-1),));k=F.rms_norm(k,(k.size(-1),));cos,sin=self.rotary(seqlen,xq.device,q.dtype);q=apply_rotary_emb(q,cos,sin,self.rope_dims);k=apply_rotary_emb(k,cos,sin,self.rope_dims);q=q*self.q_gain.to(dtype=q.dtype)[None,None,:,None]
		if self.use_dcmha and self.attn_window_size is not None:
			if self.num_kv_heads!=self.num_heads:
				k=k[:,:,:,None,:].expand(bsz,seqlen,self.num_kv_heads,self.num_heads//self.num_kv_heads,self.head_dim).reshape(bsz,seqlen,self.num_heads,self.head_dim)
				v=v[:,:,:,None,:].expand(bsz,seqlen,self.num_kv_heads,self.num_heads//self.num_kv_heads,self.head_dim).reshape(bsz,seqlen,self.num_heads,self.head_dim)
			q=rearrange(q,'b s h d -> b h s d');k=rearrange(k,'b s h d -> b h s d');v=rearrange(v,'b s h d -> b h s d')
			if self._cached_window_mask is None or self._cached_window_mask.shape[-1]!=seqlen:self._cached_window_mask=make_window_mask(seqlen,self.attn_window_size).to(q.device)
			mask=self._cached_window_mask
			project_logits=True;project_probs=True;pre_proj_dw_args,post_proj_dw_args,_=self.dyn_w_proj(xq)
			(pre_qw1,pre_qw2,pre_kw1,pre_kw2,pre_qdd,pre_kdd)=pre_proj_dw_args;(post_qw1,post_qw2,post_kw1,post_kw2,post_qdd,post_kdd)=post_proj_dw_args
			pre_sw,post_sw=None,None
			y=torch.zeros(bsz,self.num_heads,seqlen,self.head_dim).to(q.device,dtype=q.dtype);window_size=seqlen if self.attn_window_size is None else self.attn_window_size;q=q*self.head_dim**-0.5
			q_chunk_size=128;chunk_num=seqlen//q_chunk_size if seqlen%q_chunk_size==0 else seqlen//q_chunk_size+1
			for i in range(chunk_num):
				start,stop=i*q_chunk_size,(i+1)*q_chunk_size;stop=min(stop,seqlen);kv_start=max(0,stop-q_chunk_size-window_size)
				_q=q[:,:,start:stop,:];_k,_v=k[:,:,kv_start:stop,:],v[:,:,kv_start:stop,:]
				_atten_mask=mask[:,:,start:stop,kv_start:stop]
				_pre_proj_dw_args=slice_dw(pre_sw,pre_qw1,pre_qw2,pre_kw1,pre_kw2,pre_qdd,pre_kdd,start,stop,kv_start) if project_logits else None
				_post_proj_dw_args=slice_dw(post_sw,post_qw1,post_qw2,post_kw1,post_kw2,post_qdd,post_kdd,start,stop,kv_start) if project_probs else None
				_o=_atten_context(_q,_k,_v,_atten_mask,_pre_proj_dw_args,_post_proj_dw_args,dtype=q.dtype);y[:,:,start:stop]=_o
			y=rearrange(y,'b h s d -> b s h d')
		else:
			y=flash_attn_3_func(q,k,v,causal=True)
			if self.use_xsa:y=self._xsa_efficient(y,v)
		y=y.reshape(bsz,seqlen,dim);return self.proj(y)
class MLP(nn.Module):
	def __init__(self,dim,mlp_mult):super().__init__();hidden=int(mlp_mult*dim);self.fc=CastedLinear(dim,hidden,bias=False);self.proj=CastedLinear(hidden,dim,bias=False);self.proj._zero_init=True
	def forward(self,x):return self.proj(F.leaky_relu(self.fc(x),negative_slope=.5).square())
class Block(nn.Module):
	def __init__(self,dim,num_heads,num_kv_heads,mlp_mult,rope_base,qk_gain_init,train_seq_len,layer_idx=0,ln_scale=False,use_kv_shift=False,use_mudd=False, attn_window_size=None,use_dcmha=False):super().__init__();self.attn_norm=RMSNorm();self.mlp_norm=RMSNorm();self.attn=CausalSelfAttention(dim,num_heads,num_kv_heads,rope_base,qk_gain_init,train_seq_len,use_kv_shift=use_kv_shift,attn_window_size=attn_window_size,use_dcmha=use_dcmha);self.mlp=MLP(dim,mlp_mult);self.attn_scale=nn.Parameter(torch.ones(dim,dtype=torch.float32));self.mlp_scale=nn.Parameter(torch.ones(dim,dtype=torch.float32));self.resid_mix=nn.Parameter(torch.stack((torch.ones(dim),torch.zeros(dim))).float()) if not use_mudd else None;self.ln_scale_factor=1./math.sqrt(layer_idx+1)if ln_scale else 1.;self.parallel=False;self.use_mudd= use_mudd
	def forward(self,x,x0):
		if self.use_mudd:
			if isinstance(x, tuple):
				x_in = x[-1]
				normed_x = tuple([self.attn_norm(x_i) * self.ln_scale_factor for x_i in x[:-1]])
			else:
				x_in = x
				normed_x = self.attn_norm(x_in) * self.ln_scale_factor
		else:
			mix=self.resid_mix.to(dtype=x.dtype)
			x_in=mix[0][None,None,:]*x+mix[1][None,None,:]*x0
			normed_x = self.attn_norm(x_in)*self.ln_scale_factor
		attn_out=self.attn(normed_x)
		if self.parallel:mlp_out=self.mlp(self.mlp_norm(x_in)*self.ln_scale_factor);x_out=x_in+self.attn_scale.to(dtype=x_in.dtype)[None,None,:]*attn_out+self.mlp_scale.to(dtype=x_in.dtype)[None,None,:]*mlp_out
		else:x_out=x_in+self.attn_scale.to(dtype=x_in.dtype)[None,None,:]*attn_out;x_out=x_out+self.mlp_scale.to(dtype=x_out.dtype)[None,None,:]*self.mlp(self.mlp_norm(x_out)*self.ln_scale_factor)
		return x_out
class GPT(nn.Module):
	def __init__(self,h):
		super().__init__()
		if h.logit_softcap<=.0:raise ValueError(f"logit_softcap must be positive, got {h.logit_softcap}")
		self.tie_embeddings=h.tie_embeddings;self.tied_embed_init_std=h.tied_embed_init_std;self.logit_softcap=h.logit_softcap;self.tok_emb=nn.Embedding(h.vocab_size,h.embedding_dim)
		if h.embedding_dim!=h.model_dim:self.embed_proj=CastedLinear(h.embedding_dim,h.model_dim,bias=False);self.head_proj=CastedLinear(h.model_dim,h.embedding_dim,bias=False)
		else:self.embed_proj=None;self.head_proj=None
		self.use_mudd=h.use_mudd;self.use_kv_shift=h.use_kv_shift;self.use_dcmha=h.use_dcmha;self.num_layers=h.num_layers
		self.num_encoder_layers=h.num_layers//2;self.num_decoder_layers=h.num_layers-self.num_encoder_layers
		attn_window_sizes=[None]*h.num_layers
		if h.use_dcmha:attn_window_sizes=[256,None,256,256]*3
		self.blocks=nn.ModuleList([Block(h.model_dim,h.num_heads,h.num_kv_heads,h.mlp_mult,h.rope_base,h.qk_gain_init,h.train_seq_len,layer_idx=i,ln_scale=h.ln_scale,use_kv_shift=h.use_kv_shift,use_mudd=h.use_mudd, attn_window_size=attn_window_sizes[i],use_dcmha=h.use_dcmha and attn_window_sizes[i] is not None)for i in range(h.num_layers)])
		if h.rope_dims>0:
			head_dim=h.model_dim//h.num_heads
			for block in self.blocks:block.attn.rope_dims=h.rope_dims;block.attn.rotary=Rotary(head_dim,base=h.rope_base,train_seq_len=h.train_seq_len,rope_dims=h.rope_dims)
		self.final_norm=RMSNorm();self.lm_head=None if h.tie_embeddings else CastedLinear(h.embedding_dim,h.vocab_size,bias=False)
		if self.lm_head is not None:self.lm_head._zero_init=True
		if h.xsa_last_n>0:
			for i in range(max(0,h.num_layers-h.xsa_last_n),h.num_layers):self.blocks[i].attn.use_xsa=True
		if h.parallel_residual_start>=0:
			for i in range(h.parallel_residual_start,h.num_layers):self.blocks[i].parallel=True
		if self.use_mudd:
			self.looping_active=True
		else:
			self.looping_active=False
		if h.num_loops>0:
			loop_seg=list(range(h.loop_start,h.loop_end+1));all_indices=list(range(h.loop_start))
			for _ in range(h.num_loops+1):all_indices.extend(loop_seg)
			all_indices.extend(range(h.loop_end+1,h.num_layers));num_enc=len(all_indices)//2;self.encoder_indices=all_indices[:num_enc];self.decoder_indices=all_indices[num_enc:]
		else:self.encoder_indices=list(range(self.num_encoder_layers));self.decoder_indices=list(range(self.num_encoder_layers,h.num_layers))
		self.num_skip_weights=min(len(self.encoder_indices),len(self.decoder_indices))
		self.skip_weights=nn.Parameter(torch.ones(self.num_skip_weights,h.model_dim,dtype=torch.float32)) if not h.use_mudd else None
		self.skip_gates=nn.Parameter(torch.zeros(self.num_skip_weights,h.model_dim,dtype=torch.float32)) if h.skip_gates_enabled and not h.use_mudd else None
		if h.use_mudd:
			looped_num_layers = len(all_indices)
			self.num_ways = [4] * looped_num_layers
			self.mudd_q_dilation=h.mudd_q_dilation;self.mudd_k_dilation=h.mudd_k_dilation;self.mudd_in_channels=h.model_dim
			local_window_sizes=[None]*looped_num_layers;num_base_layers=1
			self.dynamic_dense=nn.ModuleList([MultiwayDynamicDenseBlock(h.model_dim,i,last_layer=i==looped_num_layers-1,multiway=True,q_dilation=self.mudd_q_dilation,k_dilation=self.mudd_k_dilation,base_layer=num_base_layers,mudd_in_channels=self.mudd_in_channels,num_ways=self.num_ways[i],local_window_size=local_window_sizes[i]) if i%self.mudd_q_dilation==0 or i==looped_num_layers-1 else None for i in range(looped_num_layers)])
		else:self.dynamic_dense=nn.ModuleList()
		if h.mudd_prenorm:self.mudd_prenorm=RMSNorm()
		else:self.mudd_prenorm=lambda x:x
		self._init_weights()
	def _init_weights(self):
		if self.tie_embeddings:nn.init.normal_(self.tok_emb.weight,mean=.0,std=self.tied_embed_init_std)
		for(name,module)in self.named_modules():
			if isinstance(module,nn.Linear):
				if getattr(module,'_zero_init',False):nn.init.zeros_(module.weight)
				elif'dynamic_dense'in name:nn.init.normal_(module.weight,mean=0.0,std=0.006)
				elif'kv_shift'in name:pass
				elif module.weight.ndim==2 and module.weight.shape[0]>=64 and module.weight.shape[1]>=64:nn.init.orthogonal_(module.weight,gain=1.)
	def forward_logits(self,input_ids):
		x=self.tok_emb(input_ids);x=F.rms_norm(x,(x.size(-1),))
		if self.embed_proj is not None:x=self.embed_proj(x)
		x0=x;hiddens=[];skips=[];enc_iter=self.encoder_indices if self.looping_active else range(self.num_encoder_layers);dec_iter=self.decoder_indices if self.looping_active else range(self.num_encoder_layers,self.num_encoder_layers+self.num_decoder_layers)
		if self.use_mudd:hiddens.append(self.mudd_prenorm(x))
		mudd_idx=0;looped_num_layers=len(enc_iter)+len(dec_iter) if self.use_mudd else 0
		for i in enc_iter:
			x=self.blocks[i](x,x0)
			if not self.use_mudd:
				skips.append(x)
			if self.use_mudd:
				if mudd_idx%self.mudd_k_dilation==0:hiddens.append(self.mudd_prenorm(x))
				if mudd_idx%self.mudd_q_dilation==0:
					dw=self.dynamic_dense[mudd_idx](x);mixed=self.dynamic_dense[mudd_idx].layer_mix(x,hiddens,dw);x=mixed
				mudd_idx+=1
		for(skip_idx,i)in enumerate(dec_iter):
			if not self.use_mudd and skip_idx<self.num_skip_weights and skips:
				scaled_skip=self.skip_weights[skip_idx].to(dtype=x.dtype)[None,None,:]*skips.pop()
				if self.skip_gates is not None:g=torch.sigmoid(self.skip_gates[skip_idx].to(dtype=x.dtype))[None,None,:];x=torch.lerp(scaled_skip,x,g)
				else:x=x+scaled_skip
			x=self.blocks[i](x,x0)
			if self.use_mudd:
				if mudd_idx%self.mudd_k_dilation==0:hiddens.append(self.mudd_prenorm(x))
				if mudd_idx%self.mudd_q_dilation==0 or mudd_idx==looped_num_layers-1:
					dw=self.dynamic_dense[mudd_idx](x);mixed=self.dynamic_dense[mudd_idx].layer_mix(x,hiddens,dw);x=mixed
				mudd_idx+=1
		x=self.final_norm(x)
		if self.head_proj is not None:x=self.head_proj(x)
		if self.tie_embeddings:logits_proj=F.linear(x,self.tok_emb.weight)
		else:logits_proj=self.lm_head(x)
		return self.logit_softcap*torch.tanh(logits_proj/self.logit_softcap)
	def forward(self,input_ids,target_ids):logits=self.forward_logits(input_ids);return F.cross_entropy(logits.reshape(-1,logits.size(-1)).float(),target_ids.reshape(-1),reduction='mean')
def classify_param(name):
	if'tok_emb'in name or'lm_head'in name:return'embed'
	if'.mlp.'in name:return'mlp'
	if'.attn.'in name or'.proj.'in name and'.mlp.'not in name:return'attn'
	return'other'
@torch.compile
def zeropower_via_newtonschulz5(G,steps=10,eps=1e-07):
	a,b,c=3.4445,-4.775,2.0315;X=G.bfloat16();X/=X.norm()+eps;transposed=G.size(0)>G.size(1)
	if transposed:X=X.T
	for _ in range(steps):A=X@X.T;B=b*A+c*A@A;X=a*X+B@X
	return X.T if transposed else X
class Muon(torch.optim.Optimizer):
	def __init__(self,params,lr,momentum,backend_steps,nesterov=True,weight_decay=.0,row_normalize=False):super().__init__(params,dict(lr=lr,momentum=momentum,backend_steps=backend_steps,nesterov=nesterov,weight_decay=weight_decay,row_normalize=row_normalize))
	@torch.no_grad()
	def step(self,closure=None):
		loss=None
		if closure is not None:
			with torch.enable_grad():loss=closure()
		distributed=dist.is_available()and dist.is_initialized();world_size=dist.get_world_size()if distributed else 1;rank=dist.get_rank()if distributed else 0
		for group in self.param_groups:
			params=group['params']
			if not params:continue
			lr=group['lr'];momentum=group['momentum'];backend_steps=group['backend_steps'];nesterov=group['nesterov'];total_params=sum(int(p.numel())for p in params);updates_flat=torch.zeros(total_params,device=params[0].device,dtype=torch.bfloat16);curr=0
			for(i,p)in enumerate(params):
				if i%world_size==rank and p.grad is not None:
					g=p.grad;state=self.state[p]
					if'momentum_buffer'not in state:state['momentum_buffer']=torch.zeros_like(g)
					buf=state['momentum_buffer'];buf.mul_(momentum).add_(g)
					if nesterov:g=g.add(buf,alpha=momentum)
					if group.get('row_normalize',False):row_norms=g.float().norm(dim=-1,keepdim=True).clamp_min(1e-07);g=g/row_norms.to(g.dtype)
					g=zeropower_via_newtonschulz5(g,steps=backend_steps);g*=max(1,g.size(0)/g.size(1))**.5;updates_flat[curr:curr+p.numel()]=g.reshape(-1)
				curr+=p.numel()
			if distributed:dist.all_reduce(updates_flat,op=dist.ReduceOp.SUM)
			wd=group.get('weight_decay',.0);curr=0
			for p in params:
				if wd>.0:p.data.mul_(1.-lr*wd)
				g=updates_flat[curr:curr+p.numel()].view_as(p).to(dtype=p.dtype);p.add_(g,alpha=-lr);curr+=p.numel()
		return loss
CONTROL_TENSOR_NAME_PATTERNS=tuple(pattern for pattern in os.environ.get('CONTROL_TENSOR_NAME_PATTERNS','attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,skip_gates').split(',')if pattern)
class Optimizers:
	def __init__(self,h,base_model):
		block_named_params=list(base_model.blocks.named_parameters());matrix_params=[p for(name,p)in block_named_params if p.ndim==2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)];scalar_params=[p for(name,p)in block_named_params if p.ndim<2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)]
		if base_model.skip_weights is not None and base_model.skip_weights.numel()>0:scalar_params.append(base_model.skip_weights)
		if base_model.skip_gates is not None and base_model.skip_gates.numel()>0:scalar_params.append(base_model.skip_gates)
		mudd_scalar_params=[];mudd_matrix_params=[]
		if base_model.use_mudd:
			for block in base_model.dynamic_dense:
				if block is not None:mudd_scalar_params.extend([p for p in block.parameters() if p.ndim<2]);mudd_matrix_params.extend([p for p in block.parameters() if p.ndim==2])
		if base_model.use_dcmha:
			dcmha_scalar_params=[]
			for block in base_model.blocks:
				if block.attn.use_dcmha:dcmha_scalar_params.extend([p for p in block.attn.dyn_w_proj.parameters()])
			scalar_params.extend(dcmha_scalar_params)
		if base_model.use_kv_shift:
			kv_shift_params=[]
			for block in base_model.blocks:
				if block.attn.use_kv_shift:kv_shift_params.extend([block.attn.kv_shift.weight])
			scalar_params.extend(kv_shift_params)
		scalar_params.extend(mudd_scalar_params+mudd_matrix_params)
		token_lr=h.tied_embed_lr if h.tie_embeddings else h.embed_lr;tok_params=[{'params':[base_model.tok_emb.weight],'lr':token_lr,'base_lr':token_lr}];self.optimizer_tok=torch.optim.AdamW(tok_params,betas=(h.beta1,h.beta2),eps=h.adam_eps,weight_decay=h.embed_wd,fused=True);self.optimizer_muon=Muon(matrix_params,lr=h.matrix_lr,momentum=h.muon_momentum,backend_steps=h.muon_backend_steps,weight_decay=h.muon_wd,row_normalize=h.muon_row_normalize)
		for group in self.optimizer_muon.param_groups:group['base_lr']=h.matrix_lr
		self.optimizer_scalar=torch.optim.AdamW([{'params':scalar_params,'lr':h.scalar_lr,'base_lr':h.scalar_lr}],betas=(h.beta1,h.beta2),eps=h.adam_eps,weight_decay=h.adam_wd,fused=True);self.optimizers=[self.optimizer_tok,self.optimizer_muon,self.optimizer_scalar]
		if base_model.lm_head is not None:self.optimizer_head=torch.optim.Adam([{'params':[base_model.lm_head.weight],'lr':h.head_lr,'base_lr':h.head_lr}],betas=(h.beta1,h.beta2),eps=h.adam_eps,fused=True);self.optimizers.insert(1,self.optimizer_head)
		else:self.optimizer_head=None
	def __iter__(self):return iter(self.optimizers)
	def zero_grad_all(self):
		for opt in self.optimizers:opt.zero_grad(set_to_none=True)
	def step(self):
		for opt in self.optimizers:opt.step()
		self.zero_grad_all()
def restore_fp32_params(model):
	for module in model.modules():
		if isinstance(module,CastedLinear):module.float()
	for(name,param)in model.named_parameters():
		if(param.ndim<2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS))and param.dtype!=torch.float32:param.data=param.data.float()
def collect_hessians(model,train_loader,h,device,n_calibration_batches=64):
	hessians={};hooks=[]
	def make_hook(name):
		def hook_fn(module,inp,out):
			x=inp[0].detach().float()
			if x.ndim==3:x=x.reshape(-1,x.shape[-1])
			if name not in hessians:hessians[name]=torch.zeros(x.shape[1],x.shape[1],dtype=torch.float32,device=device)
			hessians[name].addmm_(x.T,x)
		return hook_fn
	for(name,module)in model.named_modules():
		if isinstance(module,CastedLinear)and module.weight.numel()>65536:
			cat=classify_param(name+'.weight')
			if cat in('mlp','attn'):hooks.append(module.register_forward_hook(make_hook(name+'.weight')))
	if model.tie_embeddings:
		hook_module=model.head_proj if model.head_proj is not None else model.final_norm
		def make_output_hook(name):
			def hook_fn(module,inp,out):
				x=out.detach().float()
				if x.ndim==3:x=x.reshape(-1,x.shape[-1])
				if name not in hessians:hessians[name]=torch.zeros(x.shape[1],x.shape[1],dtype=torch.float32,device=device)
				hessians[name].addmm_(x.T,x)
			return hook_fn
		hooks.append(hook_module.register_forward_hook(make_output_hook('tok_emb.weight')))
	model.eval()
	with torch.no_grad():
		for _ in range(n_calibration_batches):x,_=train_loader.next_batch(h.train_batch_tokens,h.grad_accum_steps);model.forward_logits(x)
	for hook in hooks:hook.remove()
	for name in hessians:hessians[name]=hessians[name].cpu()/n_calibration_batches
	return hessians
def gptq_quantize_weight(w,H,clip_sigmas=3.,clip_range=63,block_size=128):
	W_orig=w.float().clone();rows,cols=W_orig.shape;H=H.float().clone();dead=torch.diag(H)==0;H[dead,dead]=1;damp=.01*H.diag().mean();H.diagonal().add_(damp);perm=torch.argsort(H.diag(),descending=True);invperm=torch.argsort(perm);W_perm=W_orig[:,perm].clone();W_perm[:,dead[perm]]=0;H=H[perm][:,perm];Hinv=torch.cholesky_inverse(torch.linalg.cholesky(H));Hinv=torch.linalg.cholesky(Hinv,upper=True);row_std=W_orig.std(dim=1);s=(clip_sigmas*row_std/clip_range).clamp_min(1e-10).to(torch.float16);sf=s.float();Q=torch.zeros(rows,cols,dtype=torch.int8);W_work=W_perm.clone()
	for i1 in range(0,cols,block_size):
		i2=min(i1+block_size,cols);W_block=W_work[:,i1:i2].clone();Hinv_block=Hinv[i1:i2,i1:i2];Err=torch.zeros(rows,i2-i1)
		for j in range(i2-i1):w_col=W_block[:,j];d=Hinv_block[j,j];q_col=torch.clamp(torch.round(w_col/sf),-clip_range,clip_range);Q[:,i1+j]=q_col.to(torch.int8);err=(w_col-q_col.float()*sf)/d;Err[:,j]=err;W_block[:,j:]-=err.unsqueeze(1)*Hinv_block[j,j:].unsqueeze(0)
		if i2<cols:W_work[:,i2:]-=Err@Hinv[i1:i2,i2:]
	return Q[:,invperm],s
def gptq_mixed_quantize(state_dict,hessians,h):
	result={};meta={}
	for(name,tensor)in state_dict.items():
		t=tensor.detach().cpu().contiguous()
		if not t.is_floating_point()or t.numel()<=65536:result[name]=t.to(torch.float16)if t.is_floating_point()else t;meta[name]='passthrough (float16)';continue
		cs=h.embed_clip_sigmas if'tok_emb'in name else h.matrix_clip_sigmas;bits=h.embed_bits if'tok_emb'in name else h.matrix_bits;q,s=gptq_quantize_weight(t,hessians[name],clip_sigmas=cs,clip_range=2**(bits-1)-1);result[name+'.q']=q;result[name+'.scale']=s;meta[name]=f"gptq (int{bits})"
	categories=collections.defaultdict(set)
	for(name,cat)in meta.items():short=re.sub('\\.\\d+$','',re.sub('blocks\\.\\d+','blocks',name));categories[cat].add(short)
	log('Quantized weights:')
	for cat in sorted(categories):log(f"  {cat}: {", ".join(sorted(categories[cat]))}")
	return result,meta
def dequantize_mixed(result,meta,template_sd):
	out={}
	for(name,orig)in template_sd.items():
		info=meta.get(name)
		if info is None:continue
		orig_dtype=orig.dtype
		if'passthrough'in info:
			t=result[name]
			if t.dtype==torch.float16 and orig_dtype in(torch.float32,torch.bfloat16):t=t.to(orig_dtype)
			out[name]=t;continue
		q,s=result[name+'.q'],result[name+'.scale']
		if s.ndim>0:out[name]=(q.float()*s.float().view(q.shape[0],*[1]*(q.ndim-1))).to(orig_dtype)
		else:out[name]=(q.float()*float(s.item())).to(orig_dtype)
	return out
_BSHF_MAGIC=b'BSHF'
def _byte_shuffle(data,stride=2):
	if stride<=1 or len(data)<stride:return data
	src=np.frombuffer(data,dtype=np.uint8);n=len(src);out=np.empty(n,dtype=np.uint8);dest_off=0
	for pos in range(stride):chunk=src[pos::stride];out[dest_off:dest_off+len(chunk)]=chunk;dest_off+=len(chunk)
	return _BSHF_MAGIC+bytes([stride])+out.tobytes()
def _byte_unshuffle(data):
	if len(data)<5 or data[:4]!=_BSHF_MAGIC:return data
	stride=data[4]
	if stride<2:return data[5:]
	payload=np.frombuffer(data,dtype=np.uint8,offset=5);n=len(payload);out=np.empty(n,dtype=np.uint8);src_off=0
	for pos in range(stride):chunk_len=n//stride+(1 if pos<n%stride else 0);out[pos::stride][:chunk_len]=payload[src_off:src_off+chunk_len];src_off+=chunk_len
	return out.tobytes()
def _compress(data,compressor):
	data=_byte_shuffle(data)
	if compressor=='lzma':return lzma.compress(data,preset=6)
	elif compressor=='brotli':import brotli;return brotli.compress(data,quality=11)
	raise ValueError(f"Unknown compressor: {compressor!r}")
def _decompress(data,compressor):
	if compressor=='lzma':raw=lzma.decompress(data)
	elif compressor=='brotli':import brotli;raw=brotli.decompress(data)
	else:raise ValueError(f"Unknown compressor: {compressor!r}")
	raw=_byte_unshuffle(raw);return raw
def serialize(h,base_model,code):
	code_bytes=len(code.encode('utf-8'))
	if h.is_main_process:torch.save(base_model.state_dict(),h.model_path);model_bytes=os.path.getsize(h.model_path);log(f"Serialized model: {model_bytes} bytes");log(f"Code size: {code_bytes} bytes")
	sd_cpu={k:v.detach().cpu()for(k,v)in base_model.state_dict().items()};device=torch.device('cuda',h.local_rank);log('GPTQ:collecting Hessians from calibration data...');t0=time.perf_counter();calib_loader=ShuffledSequenceLoader(h,device);hessians=collect_hessians(base_model,calib_loader,h,device,n_calibration_batches=h.gptq_calibration_batches);log(f"GPTQ:collected {len(hessians)} Hessians in {time.perf_counter()-t0:.1f}s");quant_result,quant_meta=gptq_mixed_quantize(sd_cpu,hessians,h);quant_buf=io.BytesIO();torch.save({'w':quant_result,'m':quant_meta},quant_buf);quant_raw=quant_buf.getvalue();quant_blob=_compress(quant_raw,h.compressor);quant_file_bytes=len(quant_blob);bytes_total=quant_file_bytes+code_bytes
	if h.is_main_process:
		with open(h.quantized_model_path,'wb')as f:f.write(quant_blob)
		log(f"Serialized model quantized+{h.compressor}: {quant_file_bytes} bytes");log(f"Total submission size quantized+{h.compressor}: {bytes_total} bytes")
	return bytes_total,quant_file_bytes
def deserialize(h,device):
	eval_model=GPT(h).to(device).bfloat16();restore_fp32_params(eval_model);sd_cpu={k:v.detach().cpu()for(k,v)in eval_model.state_dict().items()}
	with open(h.quantized_model_path,'rb')as f:quant_blob_disk=f.read()
	quant_state=torch.load(io.BytesIO(_decompress(quant_blob_disk,h.compressor)),map_location='cpu');deq_state=dequantize_mixed(quant_state['w'],quant_state['m'],sd_cpu);eval_model.load_state_dict(deq_state,strict=True);return eval_model
def _loss_bpb(loss_sum,token_count,byte_count):val_loss=(loss_sum/token_count).item();val_bpb=val_loss/math.log(2.)*(token_count.item()/byte_count.item());return val_loss,val_bpb
def eval_val(h,device,val_data,model):
	seq_len=h.eval_seq_len;local_batch_tokens=h.val_batch_tokens//(h.world_size*h.grad_accum_steps)
	if local_batch_tokens<seq_len:raise ValueError(f"VAL_BATCH_SIZE must provide at least one sequence per rank; got VAL_BATCH_SIZE={h.val_batch_tokens}, WORLD_SIZE={h.world_size}, GRAD_ACCUM_STEPS={h.grad_accum_steps}, seq_len={seq_len}")
	local_batch_seqs=local_batch_tokens//seq_len;total_seqs=(val_data.val_tokens.numel()-1)//seq_len;seq_start=total_seqs*h.rank//h.world_size;seq_end=total_seqs*(h.rank+1)//h.world_size;val_loss_sum=torch.zeros((),device=device,dtype=torch.float64);val_token_count=torch.zeros((),device=device,dtype=torch.float64);val_byte_count=torch.zeros((),device=device,dtype=torch.float64);model.eval()
	with torch.inference_mode():
		for batch_seq_start in range(seq_start,seq_end,local_batch_seqs):
			batch_seq_end=min(batch_seq_start+local_batch_seqs,seq_end);raw_start=batch_seq_start*seq_len;raw_end=batch_seq_end*seq_len+1;local=val_data.val_tokens[raw_start:raw_end].to(device=device,dtype=torch.int64,non_blocking=True);x=local[:-1].reshape(-1,seq_len);y=local[1:].reshape(-1,seq_len)
			with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=True):batch_loss=model(x,y).detach()
			batch_token_count=float(y.numel());val_loss_sum+=batch_loss.to(torch.float64)*batch_token_count;val_token_count+=batch_token_count;prev_ids=x.reshape(-1);tgt_ids=y.reshape(-1);token_bytes=val_data.base_bytes_lut[tgt_ids].to(dtype=torch.int16);token_bytes+=(val_data.has_leading_space_lut[tgt_ids]&~val_data.is_boundary_token_lut[prev_ids]).to(dtype=torch.int16);val_byte_count+=token_bytes.to(torch.float64).sum()
	if dist.is_available()and dist.is_initialized():dist.all_reduce(val_loss_sum,op=dist.ReduceOp.SUM);dist.all_reduce(val_token_count,op=dist.ReduceOp.SUM);dist.all_reduce(val_byte_count,op=dist.ReduceOp.SUM)
	model.train();return _loss_bpb(val_loss_sum,val_token_count,val_byte_count)
def eval_val_sliding(h,device,val_data,base_model,batch_seqs=32):
	base_model.eval();logits_fn=torch.compile(base_model.forward_logits,dynamic=False,fullgraph=True);seq_len=h.eval_seq_len;context_size=seq_len-h.eval_stride;total_tokens=val_data.val_tokens.numel()-1;window_starts=[ws for ws in range(0,total_tokens,h.eval_stride)if ws+context_size<total_tokens];total_windows=len(window_starts);my_s=total_windows*h.rank//h.world_size;my_e=total_windows*(h.rank+1)//h.world_size;my_windows=window_starts[my_s:my_e];loss_sum=torch.zeros((),device=device,dtype=torch.float64);token_count=torch.zeros((),device=device,dtype=torch.float64);byte_count=torch.zeros((),device=device,dtype=torch.float64)
	with torch.inference_mode():
		for bi in range(0,len(my_windows),batch_seqs):
			batch_ws=my_windows[bi:bi+batch_seqs];bsz=len(batch_ws);x_batch=torch.zeros(bsz,seq_len,dtype=torch.int64,device=device);y_batch=torch.zeros(bsz,seq_len,dtype=torch.int64,device=device);wlens=[]
			for(i,ws)in enumerate(batch_ws):we=min(ws+seq_len,total_tokens);wlen=we-ws;wlens.append(wlen);chunk=val_data.val_tokens[ws:we+1].to(dtype=torch.int64,device=device);x_batch[i,:wlen]=chunk[:-1];y_batch[i,:wlen]=chunk[1:]
			with torch.autocast(device_type='cuda',dtype=torch.bfloat16):logits=logits_fn(x_batch)
			nll=F.cross_entropy(logits.reshape(-1,logits.size(-1)).float(),y_batch.reshape(-1),reduction='none').reshape(bsz,seq_len)
			for(i,ws)in enumerate(batch_ws):wlen=wlens[i];s=0 if ws==0 else context_size;scored_nll=nll[i,s:wlen].to(torch.float64);loss_sum+=scored_nll.sum();token_count+=float(wlen-s);tgt=y_batch[i,s:wlen];prev=x_batch[i,s:wlen];tb=val_data.base_bytes_lut[tgt].to(torch.float64);tb+=(val_data.has_leading_space_lut[tgt]&~val_data.is_boundary_token_lut[prev]).to(torch.float64);byte_count+=tb.sum()
	if dist.is_available()and dist.is_initialized():dist.all_reduce(loss_sum,op=dist.ReduceOp.SUM);dist.all_reduce(token_count,op=dist.ReduceOp.SUM);dist.all_reduce(byte_count,op=dist.ReduceOp.SUM)
	base_model.train();return _loss_bpb(loss_sum,token_count,byte_count)
def eval_val_ttt(h,device,val_data,base_model,batch_seqs=32):
	rank=h.rank;world_size=h.world_size;seq_len=h.eval_seq_len;stride=h.eval_stride;total_tokens=val_data.val_tokens.numel()-1;ttt_chunk=h.ttt_chunk_tokens;context_size=seq_len-stride;window_starts=[ws for ws in range(0,total_tokens,stride)if ws+context_size<total_tokens];num_chunks=(total_tokens+ttt_chunk-1)//ttt_chunk;chunk_windows=[[]for _ in range(num_chunks)]
	for ws in window_starts:wlen=min(ws+seq_len,total_tokens)-ws;s=0 if ws==0 else context_size;scored_start=ws+s;ci=min(scored_start//ttt_chunk,num_chunks-1);chunk_windows[ci].append(ws)
	log(f"ttt:start chunks={num_chunks} ttt_lr={h.ttt_lr} ttt_epochs={h.ttt_epochs}");compiled_logits=torch.compile(base_model.forward_logits,dynamic=False,fullgraph=True);loss_sum=torch.zeros((),device=device,dtype=torch.float64);token_count=torch.zeros((),device=device,dtype=torch.float64);byte_count=torch.zeros((),device=device,dtype=torch.float64);ttt_params=[p for p in base_model.parameters()]
	for p in ttt_params:p.requires_grad_(True)
	optimizer=torch.optim.SGD(ttt_params,lr=h.ttt_lr,momentum=h.ttt_momentum)
	for ci in range(num_chunks):
		windows=chunk_windows[ci]
		if not windows:continue
		chunk_start=ci*ttt_chunk;chunk_end=min((ci+1)*ttt_chunk,total_tokens);my_s=len(windows)*rank//world_size;my_e=len(windows)*(rank+1)//world_size;my_windows=windows[my_s:my_e];base_model.eval()
		with torch.no_grad():
			for bi in range(0,len(my_windows),batch_seqs):
				batch_ws=my_windows[bi:bi+batch_seqs];bsz=len(batch_ws);x_batch=torch.zeros(bsz,seq_len,dtype=torch.int64,device=device);y_batch=torch.zeros(bsz,seq_len,dtype=torch.int64,device=device);wlens=[]
				for(i,ws)in enumerate(batch_ws):we=min(ws+seq_len,total_tokens);wlen=we-ws;wlens.append(wlen);chunk_tok=val_data.val_tokens[ws:we+1].to(dtype=torch.int64,device=device);x_batch[i,:wlen]=chunk_tok[:-1];y_batch[i,:wlen]=chunk_tok[1:]
				with torch.autocast(device_type='cuda',dtype=torch.bfloat16):logits=compiled_logits(x_batch)
				nll=F.cross_entropy(logits.reshape(-1,logits.size(-1)).float(),y_batch.reshape(-1),reduction='none').reshape(bsz,seq_len)
				for(i,ws)in enumerate(batch_ws):wlen=wlens[i];s=0 if ws==0 else context_size;scored_nll=nll[i,s:wlen].to(torch.float64);loss_sum+=scored_nll.sum();token_count+=float(wlen-s);tgt=y_batch[i,s:wlen];prev=x_batch[i,s:wlen];tb=val_data.base_bytes_lut[tgt].to(torch.float64);tb+=(val_data.has_leading_space_lut[tgt]&~val_data.is_boundary_token_lut[prev]).to(torch.float64);byte_count+=tb.sum()
		is_last_chunk=ci==num_chunks-1
		if not is_last_chunk and h.ttt_epochs>0:
			base_model.train();chunk_seqs=(chunk_end-chunk_start)//seq_len
			if chunk_seqs>0:
				cos_lr=h.ttt_lr*.5*(1.+math.cos(math.pi*ci/max(num_chunks-1,1)))
				for pg in optimizer.param_groups:pg['lr']=cos_lr
				my_seq_s=chunk_seqs*rank//world_size;my_seq_e=chunk_seqs*(rank+1)//world_size;my_chunk_seqs=my_seq_e-my_seq_s
				for _ep in range(h.ttt_epochs):
					for bs in range(0,my_chunk_seqs,batch_seqs):
						be=min(bs+batch_seqs,my_chunk_seqs);actual_bs=my_seq_s+bs;start_tok=chunk_start+actual_bs*seq_len;end_tok=chunk_start+(my_seq_s+be)*seq_len+1
						if end_tok>val_data.val_tokens.numel():continue
						local=val_data.val_tokens[start_tok:end_tok].to(device=device,dtype=torch.int64);x=local[:-1].reshape(-1,seq_len);y=local[1:].reshape(-1,seq_len);optimizer.zero_grad(set_to_none=True)
						with torch.autocast(device_type='cuda',dtype=torch.bfloat16):loss=base_model(x,y)
						loss.backward()
						if world_size>1:
							for p in ttt_params:
								if p.grad is not None:dist.all_reduce(p.grad,op=dist.ReduceOp.AVG)
						torch.nn.utils.clip_grad_norm_(ttt_params,1.);optimizer.step()
	if dist.is_available()and dist.is_initialized():dist.all_reduce(loss_sum,op=dist.ReduceOp.SUM);dist.all_reduce(token_count,op=dist.ReduceOp.SUM);dist.all_reduce(byte_count,op=dist.ReduceOp.SUM)
	for p in base_model.parameters():p.requires_grad_(True)
	base_model.eval();return _loss_bpb(loss_sum,token_count,byte_count)
def timed_eval(label,fn,*args,**kwargs):torch.cuda.synchronize();t0=time.perf_counter();val_loss,val_bpb=fn(*args,**kwargs);torch.cuda.synchronize();elapsed_ms=1e3*(time.perf_counter()-t0);log(f"{label} val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f} eval_time:{elapsed_ms:.0f}ms");return val_loss,val_bpb
def train_model(h,device,val_data):
	base_model=GPT(h).to(device).bfloat16();restore_fp32_params(base_model)
	compiled_model=torch.compile(base_model,dynamic=False,fullgraph=True)
	# compiled_model = base_model
	if h.distributed:model=DDP(compiled_model,device_ids=[h.local_rank],broadcast_buffers=False,find_unused_parameters=False)
	else:model=compiled_model
	if h.is_main_process:
		print('model parameters:')
		for name,param in model.named_parameters():
			print(name, param.shape, param.dtype, param.mean().item(), param.std().item())
	log(f"model_params:{sum(p.numel()for p in base_model.parameters())}");optimizers=Optimizers(h,base_model);train_loader=ShuffledSequenceLoader(h,device);tb_writer=None
	if h.tensorboard_dir and h.is_main_process: tensorboard_dir=os.path.join(h.tensorboard_dir,h.run_id);os.makedirs(tensorboard_dir,exist_ok=True);tb_writer=SummaryWriter(log_dir=tensorboard_dir)
	max_wallclock_ms=1e3*h.max_wallclock_seconds if h.max_wallclock_seconds>0 else None
	if max_wallclock_ms is not None:max_wallclock_ms-=h.gptq_reserve_seconds*1e3;log(f"gptq:reserving {h.gptq_reserve_seconds:.0f}s, effective={max_wallclock_ms:.0f}ms")
	def training_frac(step,elapsed_ms):
		if max_wallclock_ms is None:return step/max(h.iterations,1)
		return elapsed_ms/max(max_wallclock_ms,1e-09)
	def lr_mul(frac):
		if h.warmdown_frac<=0:return 1.
		if frac>=1.-h.warmdown_frac:return max((1.-frac)/h.warmdown_frac,h.min_lr)
		return 1.
	def step_fn(step,lr_scale):
		optimizers.zero_grad_all();train_loss=torch.zeros((),device=device)
		for micro_step in range(h.grad_accum_steps):
			if h.distributed:model.require_backward_grad_sync=micro_step==h.grad_accum_steps-1
			x,y=train_loader.next_batch(h.train_batch_tokens,h.grad_accum_steps)
			with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=True):loss=model(x,y)
			train_loss+=loss.detach();(loss/h.grad_accum_steps).backward()
		train_loss/=h.grad_accum_steps;frac=min(step/h.muon_momentum_warmup_steps,1.)if h.muon_momentum_warmup_steps>0 else 1.;muon_momentum=(1-frac)*h.muon_momentum_warmup_start+frac*h.muon_momentum
		for group in optimizers.optimizer_muon.param_groups:group['momentum']=muon_momentum
		for opt in optimizers:
			for group in opt.param_groups:group['lr']=group['base_lr']*lr_scale
		if h.grad_clip_norm>0:raw_grad_norm=torch.nn.utils.clip_grad_norm_(base_model.parameters(),h.grad_clip_norm)
		cur_lr=float(optimizers.optimizer_muon.param_groups[0]['lr'])
		optimizers.step()
		return train_loss,raw_grad_norm,cur_lr
	if h.warmup_steps>0:
		initial_model_state={name:tensor.detach().cpu().clone()for(name,tensor)in base_model.state_dict().items()};initial_optimizer_states=[copy.deepcopy(opt.state_dict())for opt in optimizers];model.train()
		for warmup_step in range(h.warmup_steps):
			_,_,_=step_fn(warmup_step,1.)
			if warmup_step<=5 or(warmup_step+1)%10==0 or warmup_step+1==h.warmup_steps:log(f"warmup_step: {warmup_step+1}/{h.warmup_steps}")
		if h.num_loops>0 and not h.use_mudd:
			base_model.looping_active=True;log(f"loop_warmup:enabled encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}")
			for warmup_step in range(h.warmup_steps):
				_,_,_=step_fn(warmup_step,1.)
				if warmup_step<=5 or(warmup_step+1)%10==0 or warmup_step+1==h.warmup_steps:log(f"loop_warmup_step: {warmup_step+1}/{h.warmup_steps}")
			if not base_model.use_mudd:base_model.looping_active=False
		base_model.load_state_dict(initial_model_state,strict=True)
		for(opt,state)in zip(optimizers,initial_optimizer_states,strict=True):opt.load_state_dict(state)
		optimizers.zero_grad_all()
		if h.distributed:model.require_backward_grad_sync=True
		train_loader=ShuffledSequenceLoader(h,device)
	ema_state={name:t.detach().float().clone()for(name,t)in base_model.state_dict().items()};ema_decay=h.ema_decay;training_time_ms=.0;stop_after_step=None;torch.cuda.synchronize();t0=time.perf_counter();step=0
	while True:
		last_step=step==h.iterations or stop_after_step is not None and step>=stop_after_step;should_validate=last_step or h.val_loss_every>0 and step%h.val_loss_every==0
		if should_validate:
			torch.cuda.synchronize();training_time_ms+=1e3*(time.perf_counter()-t0);val_loss,val_bpb=eval_val(h,device,val_data,model);log(f"{step}/{h.iterations} val_loss: {val_loss:.4f} val_bpb: {val_bpb:.4f}")
			if tb_writer is not None:tb_writer.add_scalar('val/loss',val_loss,step);tb_writer.add_scalar('val/bpb',val_bpb,step)
			torch.cuda.synchronize();t0=time.perf_counter()
		if last_step:
			if stop_after_step is not None and step<h.iterations:log(f"stopping_early: wallclock_cap train_time: {training_time_ms:.0f}ms step: {step}/{h.iterations}")
			break
		elapsed_ms=training_time_ms+1e3*(time.perf_counter()-t0);frac=training_frac(step,elapsed_ms);scale=lr_mul(frac)
		if h.num_loops>0 and not base_model.looping_active and frac>=h.enable_looping_at:base_model.looping_active=True;log(f"layer_loop:enabled step:{step} frac:{frac:.3f} encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}")
		train_loss,raw_grad_norm,cur_lr=step_fn(step,scale)
		with torch.no_grad():
			for(name,t)in base_model.state_dict().items():ema_state[name].mul_(ema_decay).add_(t.detach().float(),alpha=1.-ema_decay)
		step+=1;approx_training_time_ms=training_time_ms+1e3*(time.perf_counter()-t0);should_log_train=h.train_log_every>0 and(step<=5 or step%h.train_log_every==0 or stop_after_step is not None)
		if should_log_train:tok_per_sec=step*h.train_batch_tokens/(approx_training_time_ms/1e3);log(f"{step}/{h.iterations} train_loss: {train_loss.item():.4f} train_time: {approx_training_time_ms/60000:.1f}m, step_avg: {approx_training_time_ms/step:.2f}ms, raw_grad_norm: {raw_grad_norm:.4f}, tok/s: {tok_per_sec:.0f}")
		if tb_writer is not None and should_log_train:tb_writer.add_scalar('train/loss',train_loss.item(),step);tb_writer.add_scalar('train/raw_grad_norm',float(raw_grad_norm),step);tb_writer.add_scalar('train/learning_rate',cur_lr,step); tb_writer.add_scalar('perf/step_avg_ms',approx_training_time_ms/step,step)
		reached_cap=max_wallclock_ms is not None and approx_training_time_ms>=max_wallclock_ms
		if h.distributed and max_wallclock_ms is not None:reached_cap_tensor=torch.tensor(int(reached_cap),device=device);dist.all_reduce(reached_cap_tensor,op=dist.ReduceOp.MAX);reached_cap=bool(reached_cap_tensor.item())
		if stop_after_step is None and reached_cap:stop_after_step=step
	if tb_writer is not None:tb_writer.flush();tb_writer.close()
	log(f"peak memory allocated: {torch.cuda.max_memory_allocated()//1024//1024} MiB reserved: {torch.cuda.max_memory_reserved()//1024//1024} MiB");log('ema:applying EMA weights');current_state=base_model.state_dict();avg_state={name:t.to(dtype=current_state[name].dtype)for(name,t)in ema_state.items()};base_model.load_state_dict(avg_state,strict=True);return base_model,compiled_model
def train_and_eval(h,device):
	random.seed(h.seed);np.random.seed(h.seed);torch.manual_seed(h.seed);torch.cuda.manual_seed_all(h.seed);val_data=ValidationData(h,device);log(f"train_shards: {len(list(Path(h.datasets_dir).resolve().glob("fineweb_train_*.bin")))}");log(f"val_tokens: {val_data.val_tokens.numel()-1}");base_model,compiled_model=train_model(h,device,val_data);torch._dynamo.reset();timed_eval('pre-quantization post-ema',eval_val,h,device,val_data,compiled_model);serialize(h,base_model,Path(__file__).read_text(encoding='utf-8'))
	if h.distributed:dist.barrier()
	eval_model=deserialize(h,device)
	if h.num_loops>0:eval_model.looping_active=True
	compiled_model=torch.compile(eval_model,dynamic=False,fullgraph=True);timed_eval('quantized',eval_val,h,device,val_data,compiled_model)
	if h.sliding_window_enabled:timed_eval('quantized_sliding_window',eval_val_sliding,h,device,val_data,eval_model)
	if h.ttt_enabled and h.sliding_window_enabled:
		del eval_model,compiled_model;torch._dynamo.reset();torch.cuda.empty_cache();ttt_model=deserialize(h,device)
		if h.num_loops>0:ttt_model.looping_active=True
		timed_eval('quantized_ttt',eval_val_ttt,h,device,val_data,ttt_model);del ttt_model
	if h.etlb_enabled and h.sliding_window_enabled:
		if'eval_model'not in dir():
			eval_model=deserialize(h,device)
			if h.num_loops>0:eval_model.looping_active=True
		timed_eval('quantized_sliding_etlb',eval_val_sliding_etlb,h,device,val_data,eval_model)
def main():
	world_size=int(os.environ.get('WORLD_SIZE','1'));local_rank=int(os.environ.get('LOCAL_RANK','0'));distributed='RANK'in os.environ and'WORLD_SIZE'in os.environ
	if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
	if world_size<=0:raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
	if 8%world_size!=0:raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
	device=torch.device('cuda',local_rank);torch.cuda.set_device(device)
	if distributed:dist.init_process_group(backend='nccl',device_id=device);dist.barrier()
	torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;torch.set_float32_matmul_precision('high');from torch.backends.cuda import enable_cudnn_sdp,enable_flash_sdp,enable_math_sdp,enable_mem_efficient_sdp;enable_cudnn_sdp(False);enable_flash_sdp(True);enable_mem_efficient_sdp(False);enable_math_sdp(False);torch._dynamo.config.optimize_ddp=False;h=Hyperparameters();set_logging_hparams(h)
	if h.is_main_process:
		os.makedirs('logs',exist_ok=True);log(100*'=',console=False);log('Hyperparameters:',console=True)
		for(k,v)in sorted(vars(type(h)).items()):
			if not k.startswith('_'):log(f"  {k}: {v}",console=True)
		log('='*100,console=False);log(f"Running Python {sys.version}",console=False);log(f"Running PyTorch {torch.__version__}",console=False);log(subprocess.run(['nvidia-smi'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=False).stdout,console=False);log('='*100,console=False)
	train_and_eval(h,device)
	if distributed:dist.destroy_process_group()
if __name__=='__main__':main()
