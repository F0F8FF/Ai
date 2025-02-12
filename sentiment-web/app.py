from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import torch

app = Flask(__name__)

# 감정 분석 모델 초기화
classifier = pipeline(
    "sentiment-analysis",
    model="sangrimlee/bert-base-multilingual-cased-nsmc",
    device=0 if torch.cuda.is_available() else -1
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        text = request.json['text']
        result = classifier(text)[0]
        return jsonify({
            'sentiment': '긍정' if result['label'] == 'positive' else '부정',
            'score': f"{result['score']:.2%}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=8000, debug=True) 