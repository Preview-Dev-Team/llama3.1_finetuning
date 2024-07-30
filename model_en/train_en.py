import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    AutoConfig
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import os

# CUDA 설정 확인
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
if torch.cuda.is_available():
    print("CUDA is available. GPU can be used.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

# Hugging Face 인증 토큰 설정
hf_token = "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"

# 모델 및 데이터 설정
base_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Llama 3.1 모델
dataset_path = "train_data_en.json"  # 한국어 데이터셋 경로 train_data_en.json
new_model = "llama_en"  # 새로운 모델 이름
checkpoint_dir = "checkpoints_llama_en/"  # 체크포인트 저장 경로
os.makedirs(checkpoint_dir, exist_ok=True)

# 데이터셋 로드
dataset = load_dataset("json", data_files={"train": dataset_path}, split="train", encoding='utf-8-sig')
dataset = dataset.select(range(500))  # 데이터셋의 처음 500개 샘플 선택
print(len(dataset))
print(dataset[0])

# QLoRA config 설정
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

# 체크포인트 번호 초기화
checkpoint_number = 0
if os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir):
    checkpoint_number = max(int(d.split('-')[-1]) for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint'))

# 모델 로드 (처음에는 base_model에서 로드)
if not os.path.exists(checkpoint_dir) or not os.listdir(checkpoint_dir):
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        use_auth_token=hf_token,
        device_map="auto",
        config=AutoConfig.from_pretrained(base_model, rope_scaling={"type": "linear", "factor": 2.0})
    )
else:
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{checkpoint_number}")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        quantization_config=quant_config,
        device_map="auto"
    )

model.config.use_cache = False
model.config.pretraining_tp = 1

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    trust_remote_code=True,
    use_auth_token=hf_token
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# PEFT 설정
peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM" #"QUESTION_ANS",
)

# 모델을 PEFT 모델로 변환
model = get_peft_model(model, peft_params)

# 학습 파라미터 설정
training_params = TrainingArguments(
    output_dir="results_en/",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    warmup_steps=100,
    learning_rate=1e-5,
    fp16=False,
    bf16=True,
    logging_steps=100,
    push_to_hub=False,
    report_to='none',
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    max_grad_norm=None,
    fp16_full_eval=True,
)

# 트레이너 설정
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=256,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

# 모델 학습
trainer.train()

# 체크포인트 저장
checkpoint_number += 1  # 체크포인트 번호 증가
trainer.save_model(os.path.join(checkpoint_dir, f"checkpoint-{checkpoint_number}"))

# 최종 모델 저장
model.save_pretrained(new_model)
tokenizer.save_pretrained(new_model)

print("Training completed and model saved successfully!")