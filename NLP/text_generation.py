from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(prompt, max_length=100):
    # 토크나이저와 모델 로드
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    # 입력 텍스트 토큰화
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    # 텍스트 생성
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        do_sample=True
    )
    
    # 생성된 텍스트 디코딩
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# 다양한 프롬프트로 테스트
prompts = [
    "The future of artificial intelligence is",
    "Once upon a time in a digital world",
    "The best way to learn programming is",
    "In the next ten years, technology will"
]

print("텍스트 생성 시작...\n")

for prompt in prompts:
    print(f"프롬프트: {prompt}")
    generated = generate_text(prompt)
    print(f"생성된 텍스트: {generated}\n")
    print("-" * 80 + "\n")
