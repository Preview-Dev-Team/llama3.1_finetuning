import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer
import os
from huggingface_hub import login

login(token="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")
# 상위 폴더 경로 가져옴
parent_dir = os.path.dirname(os.getcwd())
# CUDA 설정 확인
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
if torch.cuda.is_available():
    print("CUDA is available. GPU can be used.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

# 모델 및 데이터 설정
base_model = "beomi/Llama-3-Open-Ko-8B"  # beomi님의 Llama3 한국어 파인튜닝 모델
dataset_path = "our_train_data.json"  # 데이터셋 경로
checkpoint_dir = "checkpoints/"  # 체크포인트 저장 경로
'''ai 허브 데이터로 학습시킨 걸 가져와서 우리 데이터로 추가학습'''
checkpoint_dir=os.path.join(parent_dir, "model_ko/checkpoints_ko")


# 데이터셋 로드
dataset = load_dataset("json", data_files={"train": dataset_path}, split="train", encoding='utf-8-sig')
# dataset = dataset.select(range(500))  # 데이터셋의 처음 500개 샘플 선택
print(len(dataset))
print(dataset[0]['text'])

# QLoRA config 설정
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16,
    bnb_4bit_use_double_quant=False,
)

# 체크포인트 번호 초기화 및 경로 설정
if os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir):
    checkpoint_number = max(int(d.split('-')[-1]) for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint'))
else:
    checkpoint_number=0
checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{checkpoint_number}")
print(checkpoint_path)
# 모델 로드 (첫 실행 시 base_model에서, 이후에는 체크포인트에서 로드)
if os.path.exists(checkpoint_path):
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        quantization_config=quant_config,
        device_map={"": 0}
    )
else:
    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map={"": 0}
    )

model.config.use_cache = False
model.config.pretraining_tp = 1

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# PEFT 설정
peft_params = LoraConfig(
    lora_alpha=16, # 높일수록 lora가중치의 힘이 커짐 -> 파인튜닝 영향력 커짐
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# 학습 파라미터 설정
training_params = TrainingArguments(
    output_dir="results_ko/",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    warmup_steps=100,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=100,
    push_to_hub=False,
    report_to='none',
)

# 트레이너 설정
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=256,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

# 모델 학습
trainer.train()

# 체크포인트 저장
save_checkpoint_dir = "checkpoints/"
os.makedirs(save_checkpoint_dir, exist_ok=True)
if os.path.exists(save_checkpoint_dir) and os.listdir(save_checkpoint_dir):
    checkpoint_number = max(int(d.split('-')[-1]) for d in os.listdir(save_checkpoint_dir) if d.startswith('checkpoint'))
    checkpoint_number+=1
else:
    checkpoint_number=1
trainer.save_model(os.path.join(save_checkpoint_dir, f"checkpoint-{checkpoint_number}"))

print("Training completed and model saved successfully!")