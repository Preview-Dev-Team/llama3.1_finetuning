import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig

# CUDA 메모리 비우기: GPU 메모리를 비워서 메모리 부족 문제를 예방합니다.
torch.cuda.empty_cache()

# 저장된 모델 경로 설정: 모델이 저장된 경로를 지정합니다.
model_path = "llama_ko"

# PEFT 설정 로드: 모델의 PEFT 설정을 로드합니다.
peft_config = PeftConfig.from_pretrained(model_path)

# PEFT 모델 로드: 기본 모델을 로드하고 PEFT를 적용합니다.
model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,  # 기본 모델 경로
    return_dict=True,                     # 딕셔너리 형태로 반환
    torch_dtype=torch.bfloat16,           # 모델 파라미터의 데이터 타입 설정
    device_map="auto"                     # 자동으로 디바이스 맵 설정
)
model = PeftModel.from_pretrained(model, model_path)  # PEFT 모델 로드

# 토크나이저 로드: 텍스트를 토큰으로 변환하는 토크나이저를 로드합니다.
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # 패딩 토큰이 없으면 종료 토큰을 패딩 토큰으로 설정
tokenizer.padding_side = "right"  # 패딩 토큰을 오른쪽에 추가

with torch.no_grad():  # 그래디언트 계산을 비활성화하여 메모리 사용을 줄입니다.
    # 텍스트 생성 파이프라인 생성: 모델과 토크나이저를 사용하는 파이프라인 생성
    pipe = pipeline(
        task="text-generation",  # 텍스트 생성 작업
        model=model,             # 로드한 모델
        tokenizer=tokenizer,     # 로드한 토크나이저
    )

    # 텍스트 생성 예시: 모델에 입력할 프롬프트 설정
    prompt = "Q. 주어진 리뷰의 모든 토픽과 감정을 태깅하시오. 리뷰 : 색감은 이쁜데 생각보다 무거워요."

    # 텍스트 생성 호출: 지정된 프롬프트로 텍스트를 생성합니다.
    result = pipe(
    f"[INST] {prompt} [/INST]",
    max_new_tokens=150,
    do_sample=False,
    temperature=0,
    top_p=1)

    # 결과 출력: 생성된 텍스트 출력
    print(result[0]['generated_text'])