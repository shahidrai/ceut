# ceut
 """
    Convolutional English to Urdu Translation from `"English to Urdu: Optimizing Sequence Learning in Neural Machine Translation" Please 
    fallow the setting to train the model for English to Urdu dataset Tanzil http://opus.nlpl.eu/Tanzil.php
    
    !mkdir -p checkpoints/urdu/Eng_urdu_model
!CUDA_VISIBLE_DEVICES=0 python train.py dataload/dataurdu \
    --log-interval 100 --no-progress-bar \
    --max-update 30000  --optimizer adam \
    --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt \
    --clip-norm 0.0 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --min-lr 1e-09 --update-freq 16  --keep-last-epochs 10 \
    --ddp-backend=no_c10d --max-tokens 3500 \
    --lr-scheduler cosine --warmup-init-lr 1e-7 --warmup-updates 10000 \
    --lr-shrink 1 --max-lr 0.01 --lr 1e-7 --min-lr 1e-9 --warmup-init-lr 1e-07 \
    --t-mult 1 --lr-period-updates 20000 --no-epoch-checkpoints \
    --arch conv_eng_urdu --save-dir checkpoints/urdu/Eng_urdu_model \
    --dropout 0.1
    
    """
