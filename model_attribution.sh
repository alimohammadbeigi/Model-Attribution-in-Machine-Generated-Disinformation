#export USE_ADAPTER=PET
#export ADAPTER_H=384
for rand_seed in 99; do
python mscl_domain.py \
  --embedding_size 768\
  --lr 2e-5 \
  --nepoch 4\
  --seed $rand_seed\
  --lambda_supcon 0.0\
  --batch_size 22\
  --max_length 350\
  --grad_clip_norm 10.0\
  --gradient_acc_step 1\
  --memory_bank_size 12\
  --skip_step 0\
  --warmup_step 200\
  --lambda_ce 1.0\
  --lambda_supcon 0.0\
  --m_update_interval 20\
  --lambda_adv 0.0\
  --lambda_moco 1.0\
  --hidden_dropout_prob 0.1\
  --hidden_size 256\
  --temp 0.2\
  --save_model False\


done;
