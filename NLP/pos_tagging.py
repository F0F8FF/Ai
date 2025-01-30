import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# 1. 샘플 텍스트
text = "John works at Google in New York. He loves programming and machine learning."

# 2. 문장 분리
sentences = nltk.sent_tokenize(text)
print("1) 문장 분리:", sentences)

# 3. 토큰화
tokens = nltk.word_tokenize(text)
print("\n2) 토큰화:", tokens)

# 4. 품사 태깅
pos_tags = nltk.pos_tag(tokens)
print("\n3) 품사 태깅 결과:")
for word, pos in pos_tags:
    print(f"{word}: {pos}")

# 5. 주요 품사 설명
pos_description = {
    'NNP': '고유명사',
    'NN': '일반명사',
    'VB': '동사',
    'VBZ': '3인칭 단수 동사',
    'IN': '전치사',
    'DT': '관사',
    'CC': '접속사',
    'PRP': '인칭대명사'
}

print("\n4) 주요 품사 설명:")
for word, pos in pos_tags:
    if pos in pos_description:
        print(f"{word} ({pos}): {pos_description[pos]}")
