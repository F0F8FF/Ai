from transformers import BertTokenizer, BertModel
import torch

# BERT 토크나이저와 모델 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 분석할 문장들
sentences = [
    "I love natural language processing.",
    "The weather is beautiful today.",
    "Machine learning is fascinating."
]

print("BERT 문장 분석 시작...\n")

for sentence in sentences:
    # 토큰화
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    
    # BERT 모델로 문장 분석
    outputs = model(**inputs)
    
    # 문장 임베딩 (CLS 토큰의 마지막 히든 스테이트)
    sentence_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
    
    print(f"문장: {sentence}")
    print(f"임베딩 크기: {sentence_embedding.shape}")
    print(f"임베딩 벡터 (처음 5개 값): {sentence_embedding[0][:5]}\n")

# 문장 유사도 계산
def get_sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach()

# 문장 쌍 비교
sentence_pairs = [
    ("I love programming", "I enjoy coding"),
    ("The weather is nice", "It's a beautiful day"),
    ("I love programming", "The weather is nice")
]

print("문장 유사도 분석:\n")
for sent1, sent2 in sentence_pairs:
    emb1 = get_sentence_embedding(sent1)
    emb2 = get_sentence_embedding(sent2)
    
    # 코사인 유사도 계산
    similarity = torch.nn.functional.cosine_similarity(emb1, emb2)
    
    print(f"문장 1: {sent1}")
    print(f"문장 2: {sent2}")
    print(f"유사도: {similarity.item():.4f}\n")
