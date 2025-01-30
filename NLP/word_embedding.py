# word_embedding_large.py
import gensim
from gensim.models import Word2Vec
import nltk
from nltk.corpus import brown
nltk.download('brown')

# 1. Brown 코퍼스 데이터 로드
sentences = brown.sents()
print(f"총 문장 수: {len(sentences)}")
print(f"첫 5개 문장 예시:")
for i, sent in enumerate(sentences[:5]):
    print(f"{i+1}. {' '.join(sent)}")

# 2. Word2Vec 모델 학습
model = Word2Vec(sentences, 
                vector_size=100,    # 벡터 크기
                window=5,           # 문맥 윈도우 크기
                min_count=5,        # 최소 단어 빈도
                workers=4)          # 학습 프로세스 수

print("\n학습 완료!")
print(f"총 어휘 크기: {len(model.wv.key_to_index)}")

# 3. 자주 사용되는 단어들의 유사도 확인
target_words = ['man', 'woman', 'king', 'queen', 'computer', 'book']

for word in target_words:
    if word in model.wv:
        print(f"\n'{word}'와 가장 유사한 단어들:")
        similar_words = model.wv.most_similar(word)
        for similar_word, score in similar_words:
            print(f"{similar_word}: {score:.4f}")

# 4. 단어 간 관계 분석
word_pairs = [
    ('man', 'woman'),
    ('king', 'queen'),
    ('book', 'read'),
    ('computer', 'machine')
]

print("\n단어 쌍 유사도:")
for w1, w2 in word_pairs:
    if w1 in model.wv and w2 in model.wv:
        similarity = model.wv.similarity(w1, w2)
        print(f"{w1} - {w2}: {similarity:.4f}")
