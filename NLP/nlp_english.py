import nltk
nltk.download('punkt')
nltk.download('stopwords')

# 1. 샘플 텍스트
text = "Natural Language Processing (NLP) is amazing! It helps computers understand human language."

# 2. 기본 텍스트 처리
print("원본 텍스트:", text)
print("\n1) 소문자 변환:", text.lower())
print("2) 단어 분리:", text.split())

# 3. NLTK로 토큰화
tokens = nltk.word_tokenize(text)
print("\n3) NLTK 토큰화 결과:", tokens)

# 4. 불용어(stopwords) 제거
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
words_without_stopwords = [word for word in tokens if word.lower() not in stop_words]
print("\n4) 불용어 제거 후:", words_without_stopwords)

# 5. 단어 개수 세기
from collections import Counter
word_counts = Counter(words_without_stopwords)
print("\n5) 단어 빈도:", dict(word_counts))
