epochs=100
bs=64
lr=1e-5
isUL=True
NUM_UE_MAX=1
NUM_UE_MIN=1
is_single_user=False
numencoderlayers=2
numdecoderlayers=4
n_routed_experts=31
n_activated_experts=3
n_shared_expert=1
augmentation_factor=2
d_model=64
d_model_decoder=64
d_ff=128
d_ff_d=128
SNR=20
feedback_type='random'
cr=32
scenario='null'
num_bit=5
python ./test.py  -e\
  --model_name 'WiFo_CF_base' \
  --data-dir './dataset/' \
  --scenario $scenario \
  --save-path ./checkpoints/null  \
  --pretrained ./checkpoints/xxx/last.pth  \
  --epochs $epochs \
  --d_model $d_model \
  --d_model_decoder $d_model_decoder \
  --d_ff $d_ff \
  --d_ff_d $d_ff_d \
  --num-encoder-layers $numencoderlayers \
  --num-decoder-layers $numdecoderlayers \
  --batch-size $bs \
  --lr-init $lr \
  --workers 0 \
  --cr $cr \
  --scheduler 'cosine' \
  --gpu 0 \
  --is_UL_instead $isUL \
  --NUM_UE_MAX $NUM_UE_MAX \
  --NUM_UE_MIN $NUM_UE_MIN \
  --is_single_user $is_single_user \
  --n_routed_experts $n_routed_experts \
  --n_activated_experts $n_activated_experts \
  --n_shared_expert $n_shared_expert \
  --SNR $SNR \
  --num_bit $num_bit \
  --feedback_type $feedback_type \
  --augmentation_factor $augmentation_factor



