import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig

# CUDA 메모리 비우기
torch.cuda.empty_cache()

# 저장된 모델 경로
model_path = "llama_en"

# PEFT 설정 로드
peft_config = PeftConfig.from_pretrained(model_path)

# PEFT 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    return_dict=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, model_path)

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

with torch.no_grad():
    # 파이프라인 생성
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    # 텍스트 생성 예시
    prompt = "Q. Tag topics and sentiments from a given review. Review: I bought an additional one for tabletop use, but it's a newer model and I like it because it has a rotation function. ^^ I like it because it's compact in size and doesn't take up a lot of space."

    # `max_new_tokens`를 사용하여 생성될 텍스트의 길이 설정
    result = pipe(
        f"[INST] {prompt} [/INST]",
        max_new_tokens=200,  # 생성될 최대 토큰 수 설정
        do_sample=False,
        temperature=0,
        top_p=0.9,
    )

    # 결과 출력
    print(result[0]['generated_text'])