# Llama 3.1 Instruct 파인튜닝 가이드

## 목차
1. [기본 모델 사용 예시](#기본-모델-사용-예시)
2. [모델 입력값에 따른 프롬프트 포맷팅](#모델-입력값에-따른-프롬프트-포맷팅)
3. [파인튜닝 데이터 생성 형식](#파인튜닝-데이터-생성-형식)

## 기본 모델 사용 예시 
(출처: [메타 공식 문서](https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1), [메타 허깅 페이스 레포지터리](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct))
```python
import transformers
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "당신은 상냥한 챗봇입니다."},
    {"role": "user", "content": "안녕? 너는 누구니?"},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])
# 안녕하세요? 저는 메타에서 개발한 llama3.1이에요. ~~
```

## 모델 입력값에 따른 프롬프트 포맷팅

### 프롬프트 토큰

Llama 3.1에서 사용되는 프롬프트 Special Token

| 토큰 | 설명 |
|------|------|
| <\|begin_of_text\|> | 프롬프트의 시작 |
| <\|end_of_text\|> | 프롬프트의 종료 (모델 토큰 생성 종료) |
| <\|finetune_right_pad_id\|`> | 배치에서 텍스트 시퀀스를 동일한 길이로 패딩 |
| <\|start_header_id\|> | 특정 역할을 지정하는 토큰. 가능한 역할: `[system, user, assistant, ipython]` |
| <\|eom_id\|> | 메시지 끝. 도구 호출이 필요한 실행 중지 지점 표시 |
| <\|eot_id\|> | 턴의 끝. 대화 쌍에서 한쪽의 턴이 끝남을 의미 |
| <\|python_tag\|> | 모델 응답에서 도구(파이썬) 호출을 나타내는 특별한 태그 |

위의 예제의 변환 결과

    <|begin_of_text|>

        <|start_header_id|>system<|end_header_id|>
                    당신은 상냥한 챗봇입니다.<|eot_id|>
        
        <|start_header_id|>user<|end_header_id|>
                    안녕? 너는 누구니?<|eot_id|>
        
        <|start_header_id|>assistant<|end_header_id|>
                    안녕하세요? 메타에서 개발한 llama3.1이에요. ~~ <|eot_id|>

    <|end_of_text|>
## 파인튜닝 데이터 생성 형식
```python

prompt={}
system_text = "시스템 지시문 (수정)" 
user_text = "사용자 지시문 (수정)" 
assistant_text = "모델 반환값 (target)" # 모델 응답 이후 턴 종료 및 프롬프트 종료 선언 "<|eot_id|><|end_of_text|>"

system_prompt="<|begin_of_text|><|start_header_id|>system<|end_header_id|>" + system_text + "<|eot_id|>" # 시스템 지시문 이후 턴 종료 선언 "<|eot_id|>"
user_prompt="<|start_header_id|>user<|end_header_id|>" + user_text + "<|eot_id|>" # 사용자 지시문 이후 턴 종료 선언 "<|eot_id|>"
assistant_prompt="<|start_header_id|>assistant<|end_header_id|>" + assistant_text + "<|eot_id|><|end_of_text|>" # 모델 응답 이후 턴 종료 및 프롬프트 종료 선언 "<|eot_id|><|end_of_text|>"

```