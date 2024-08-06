import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def setup_model_and_tokenizer(base_model):
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0}
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

def create_pipeline(model, tokenizer):
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

def generate_response(pipe, messages, max_new_tokens=2048):
    result = pipe(messages, max_new_tokens=max_new_tokens, temperature=0.1, top_k=10, top_p=0.9)
    return result[0]['generated_text']


if __name__ =="__main__":
    # 사용 예시
    base_model = "results"
    model, tokenizer = setup_model_and_tokenizer(base_model)
    pipe = create_pipeline(model, tokenizer)

    messages = [
        {"role": "system", "content": "당신은 리뷰를 읽고 토픽과 감정을 태깅하는 AI 어시스턴트입니다."},
        {"role": "user", "content": "다음 리뷰를 분석하세요: 디자인 정말 예쁘네요. 배송도 빠르게 왔어요!"},
    ]

    response = generate_response(pipe, messages)
    print(response)