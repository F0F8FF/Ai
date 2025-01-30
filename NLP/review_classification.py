import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import defaultdict
nltk.download('vader_lexicon')

# 감성 분석기 초기화
sia = SentimentIntensityAnalyzer()

# 샘플 리뷰 데이터
reviews = [
    "The food was amazing and the service was excellent!",
    "Terrible experience, would not recommend.",
    "Good food but slow service and expensive prices.",
    "The staff was friendly but the food was just okay.",
    "Best restaurant ever! Will definitely come back!",
    "Waited for an hour, food was cold when served.",
    "Decent place, nothing special but not bad either.",
    "Great atmosphere and reasonable prices.",
    "The quality has gone down recently.",
    "Absolutely loved everything about this place!"
]

# 리뷰 분류를 위한 딕셔너리
classified_reviews = defaultdict(list)

# 각 리뷰 분석 및 분류
print("리뷰 분석 결과:\n")

for review in reviews:
    # 감성 점수 계산
    scores = sia.polarity_scores(review)
    
    # 리뷰 분류
    if scores['compound'] >= 0.5:
        category = "매우 긍정"
    elif scores['compound'] >= 0.1:
        category = "긍정"
    elif scores['compound'] <= -0.5:
        category = "매우 부정"
    elif scores['compound'] <= -0.1:
        category = "부정"
    else:
        category = "중립"
    
    # 분류된 리뷰 저장
    classified_reviews[category].append(review)
    
    # 분석 결과 출력
    print(f"리뷰: {review}")
    print(f"감성 점수: {scores['compound']:.3f}")
    print(f"분류: {category}\n")

# 카테고리별 통계
print("\n=== 카테고리별 리뷰 수 ===")
for category, reviews in classified_reviews.items():
    print(f"{category}: {len(reviews)}개")

# 각 카테고리의 예시 리뷰 출력
print("\n=== 카테고리별 예시 리뷰 ===")
for category, reviews in classified_reviews.items():
    print(f"\n{category}:")
    for review in reviews:
        print(f"- {review}")
