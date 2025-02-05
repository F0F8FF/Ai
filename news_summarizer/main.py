from flask import Flask, request, render_template
from summarizer.extractive import NewsTextRank
from summarizer.abstractive import KoBART_Summarizer
from summarizer.utils import clean_text, evaluate_summary
import nltk
from konlpy.tag import Mecab

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def summarize():
    if request.method == 'POST':
        url = request.form.get('url', '').strip()
        news_text = request.form.get('news_text', '').strip()
        
        # 텍스트 가져오기
        if url:
            textrank = NewsTextRank()
            news_text = textrank.get_news_content(url)
        
        if news_text:
            # 텍스트 정제
            cleaned_text = clean_text(news_text)
            
            # 추출적 요약
            textrank = NewsTextRank()
            extractive_summary = textrank.summarize(cleaned_text)
            
            # 추상적 요약
            kobart = KoBART_Summarizer()
            abstractive_summary = kobart.summarize(cleaned_text)
            
            # 평가
            if abstractive_summary:
                scores = evaluate_summary(cleaned_text, abstractive_summary)
            else:
                scores = None
            
            return render_template('index.html',
                                extractive_summary=extractive_summary,
                                abstractive_summary=abstractive_summary,
                                scores=scores)
    
    return render_template('index.html')

if __name__ == '__main__':
    nltk.download('punkt')
    app.run(debug=True)