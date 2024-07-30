import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig

# Clear CUDA memory
torch.cuda.empty_cache()

# 체크포인트에서 최종 모델 로드
checkpoint_dir = "checkpoints/"
if os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir):
    checkpoint_number = max(int(d.split('-')[-1]) for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint'))
    # checkpoint_number = 1
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{checkpoint_number}")
    print(f"Loading model from checkpoint: {checkpoint_path}")
# 체크포인트가 없으면 종료
else:
    print("No checkpoints found. Please ensure the checkpoint directory exists and contains checkpoints.")
    exit()

# Load the model from checkpoint
model = AutoModelForCausalLM.from_pretrained(
    checkpoint_path,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Set the model to evaluation mode
model.eval()

# Create text generation pipeline
pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

# Example prompt
prompt = "Q. 주어진 리뷰로부터 유의미한 토픽과 토픽 점수, 감정과 감정 점수를 가진 세부 문장을 전부 추출하고, 긱 세부 문장마다 태깅하시오.\n리뷰 : 원래는 조금 더 무거운 제품을 사용했었는데, 이 제품은 엄청 가볍고 이쁘네요."
# prompt = "Q. 보조패터리에 태깅할만한 토픽들을 리스트형태로 제시하시오."
# Generate text
with torch.no_grad():
    result = pipe(
        f"<s>[INST] {prompt} [/INST]",
        max_new_tokens=150,
        do_sample=False,
        temperature=0,
        top_p=1
    )

# Print the generated text
print(result[0]['generated_text'])