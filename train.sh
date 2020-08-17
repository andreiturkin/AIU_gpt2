# partially adopted from README.md
# https://github.com/mgrankin/ru_transformers.git
#                           Apache License
#                       Version 2.0, January 2004
#                    http://www.apache.org/licenses/

export TRAIN_FILE=./data/trainData
export EVAL_FILE=./data/evaluateData

export CUDA_VISIBLE_DEVICES=0
export MODEL_SIZE=gpt2-medium
export OUTPUT=output_yt/m
export BS=3
export LR=1e-5


python run_lm_finetuning.py \
        --output_dir=$OUTPUT \
        --model_type=gpt2 \
        --model_name_or_path=$OUTPUT \
        --do_train \
        --train_data_file=$TRAIN_FILE \
        --per_gpu_train_batch_size $BS \
        --save_steps=10000 \
        --logging_steps=10 \
        --fp16 \
        --fp16_opt_level O2 \
        --warmup_samples 1600 \
        --learning_rate $LR \
        --overwrite_output_dir \
        --tokenizer_class YTEncoder \
        --tokenizer_name ./bpe/yt.model \
        --do_eval \
        --evaluate_during_training \
        --eval_steps 1000 \
        --eval_data_file=$EVAL_FILE \
        --save_total_limit 30 \
        --num_train_epochs 5.0 \
        --unfreeze_level 0\
	--block_size 50
