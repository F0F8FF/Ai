import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# 감성 분석기 초기화
sia = SentimentIntensityAnalyzer()

# 테스트할 문장들
sentences = [
    "I love this product! It's amazing.",
    "This is terrible, I hate it.",
    "The movie was good, but a bit long.",
    "The service was excellent and the staff was very friendly!",
    "I'm not sure about this, it's just okay."
]

print("감성 분석 결과:\n")

for sentence in sentences:
    # 감성 점수 계산
    scores = sia.polarity_scores(sentence)
    
    print(f"문장: {sentence}")
    print(f"긍정: {scores['pos']:.3f}")
    print(f"중립: {scores['neu']:.3f}")
    print(f"부정: {scores['neg']:.3f}")
    print(f"종합: {scores['compound']:.3f}")  # -1(매우 부정) ~ 1(매우 긍정)
    
    # 감성 판단
    if scores['compound'] >= 0.05:
        sentiment = "긍정적"
    elif scores['compound'] <= -0.05:
        sentiment = "부정적"
    else:
        sentiment = "중립적"
    
    print(f"최종 판단: {sentiment}\n")
