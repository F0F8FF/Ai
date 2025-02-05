from rouge import Rouge
import re

def clean_text(text):
    # HTML 태그 제거
    text = re.sub(r'<[^>]+>', '', text)
    # 특수문자 처리
    text = re.sub(r'[^\w\s\.]', '', text)
    # 중복 공백 제거
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def evaluate_summary(original_text, generated_summary):
    try:
        rouge = Rouge()
        scores = rouge.get_scores(generated_summary, original_text)
        
        return {
            "rouge-1": scores[0]['rouge-1']['f'],
            "rouge-2": scores[0]['rouge-2']['f'],
            "rouge-l": scores[0]['rouge-l']['f']
        }
    except Exception as e:
        print(f"Error in evaluation: {str(e)}")
        return None