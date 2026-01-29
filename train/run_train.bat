@echo off
REM ==========================================
REM GPT-2 / GPT-2-XL Perplexity Curve Training
REM Records:
REM  - Validation Perplexity vs Step
REM  - Training Time
REM  - GPU Memory Usage
REM ==========================================

python train_gpt2.py ^
  --model_name_or_path models/gpt2-xl ^
  --tokenizer_name lm_data ^
  --train_file lm_data/train.txt ^
  --validation_file lm_data/valid.txt ^
  --do_train ^
  --do_eval ^
  --block_size 256 ^
  --per_device_train_batch_size 1 ^
  --per_device_eval_batch_size 1 ^
  --gradient_accumulation_steps 16 ^
  --learning_rate 2e-5 ^
  --num_train_epochs 5 ^
  --eval_steps 50 ^
  --logging_steps 100 ^
  --save_steps 1000 ^
  --fp16 ^
  --output_dir out_gpt2_xl

pause
