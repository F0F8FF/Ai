from newspaper import Article
from konlpy.tag import Mecab
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import numpy as np
import re

class NewsTextRank:
    def __init__(self):
        try:
            self.mecab = Mecab(dicpath='/opt/homebrew/lib/mecab/dic/mecab-ko-dic')
        except:
            self.mecab = Mecab()
        
        # 뉴스 메타데이터 패턴 정의
        self.skip_patterns = [
            # 언론사 패턴
            r'연합뉴스', r'뉴시스', r'뉴스1', r'노컷뉴스', r'중앙일보', r'동아일보', 
            r'조선일보', r'한겨레', r'경향신문', r'매일경제', r'한국경제', r'서울경제',
            r'머니투데이', r'파이낸셜뉴스', r'아시아경제', r'이데일리',
            
            # 기자 및 작성자 패턴
            r'\w+\s*기자',
            r'[가-힣]{2,4}\s*(기자|팀장|특파원)',
            r'[a-zA-Z\s]+\s*correspondent',
            
            # 날짜 및 시간 패턴
            r'\d{1,2}일\s*현지시간',
            r'\(현지시간\)',
            r'\d{4}\.\d{1,2}\.\d{1,2}',
            r'\d{1,2}월\s*\d{1,2}일',
            
            # 인용 및 출처 패턴
            r'제공|자료사진|자료사진=|사진=|사진제공',
            r'출처[=:]?',
            r'저작권자?[=ⓒ©]',
            r'[=:]?\s*뉴스\s*[=:]?',
            
            # 통신사 패턴
            r'로이터', r'AP', r'AFP', r'블룸버그',
            
            # 기타 메타데이터
            r'편집자\s*(주|註)',
            r'[가-힣]+=\w+',
            r'\([^)]*제공\)',
            r'송고\s*시간?'
        ]
        
        self.skip_pattern = re.compile('|'.join(self.skip_patterns))

    def get_news_content(self, url):
        try:
            article = Article(url, language='ko')
            article.download()
            article.parse()
            return article.text
        except Exception as e:
            print(f"Error fetching news content: {str(e)}")
            return None

    def sentence_tokenize(self, text):
        # 괄호 내용 제거
        text = re.sub(r'\([^)]*(?:기자|통신|뉴스|제공|사진|시간)[^)]*\)', '', text)
        
        # 문장 분리
        sentences = re.split(r'(?<=[.!?])\s+', text)
        filtered_sentences = []
        
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 20 and not self.skip_pattern.search(sent):
                sent = re.sub(r'^\s*[\w\s]+\s*=\s*', '', sent)
                sent = re.sub(r'^\s*\d+일\s*', '', sent)
                sent = re.sub(r'^\s*\([^)]*\)\s*', '', sent)
                filtered_sentences.append(sent)
        
        return filtered_sentences

    def preprocess_sentence(self, sentence):
        # 불용어 정의
        stop_words = {
            '있다', '하다', '이다', '되다', '그', '등', '이', '가', '을', '를', '에', '의',
            '및', '또는', '그리고', '라고', '이라고', '까지', '에서', '으로', 
            '통신', '보도', '전했다', '밝혔다', '말했다', '설명했다',
            '지난해', '이번', '한편', '또한', '이어'
        }
        
        # 품사 태깅 및 단어 선택
        pos_tags = self.mecab.pos(sentence)
        words = []
        for word, pos in pos_tags:
            if (pos.startswith('NN') or pos.startswith('VV') or pos.startswith('VA')) and \
               word not in stop_words and len(word) > 1:
                words.append(word)
        
        return ' '.join(words)

    def calculate_sentence_similarity(self, tfidf_matrix):
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        np.fill_diagonal(similarity_matrix, 0)
        return similarity_matrix

    def summarize(self, text, num_sentences=3):
        if not text:
            return []
        
        sentences = self.sentence_tokenize(text)
        if len(sentences) <= num_sentences:
            return sentences
        
        preprocessed_sentences = [self.preprocess_sentence(sent) for sent in sentences]
        
        # TF-IDF 매트릭스 생성 파라미터 조정
        tfidf = TfidfVectorizer(
            min_df=1,
            max_df=0.9,
            max_features=300,
            ngram_range=(1, 2),  # 1~2단어 구문으로 조정
            token_pattern=r'(?u)\b\w+\b'  # 단어 패턴 수정
        )
        
        tfidf_matrix = tfidf.fit_transform(preprocessed_sentences)
        similarity_matrix = self.calculate_sentence_similarity(tfidf_matrix)
        
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(
            nx_graph,
            alpha=0.90,  # 감쇄 계수 증가
            max_iter=200,
            tol=1e-8
        )
        
        # 문장 선택 로직 개선
        ranked_sentences = []
        for i, sent in enumerate(sentences):
            # 위치 가중치 조정
            position_weight = 1.0
            if i == 0:  # 첫 문장
                position_weight = 1.3
            elif i < len(sentences) * 0.2:  # 앞부분 20%
                position_weight = 1.2
            elif i > len(sentences) * 0.8:  # 뒷부분 20%
                position_weight = 1.1
            
            # 문장 길이 가중치
            length_weight = min(len(sent) / 100, 1.2)  # 너무 긴 문장 페널티
            
            # 인용구 가중치
            quote_weight = 1.2 if '"' in sent or '"' in sent or '말했다' in sent else 1.0
            
            final_score = scores[i] * position_weight * length_weight * quote_weight
            ranked_sentences.append((final_score, i, sent))
        
        ranked_sentences.sort(reverse=True)
        
        # 문장 선택 시 중복 내용 확인
        selected = []
        selected_indices = set()
        
        for score, idx, sent in ranked_sentences:
            if len(selected) >= num_sentences:
                break
                
            # 이미 선택된 문장들과의 유사도 확인
            if selected:
                max_similarity = max(
                    cosine_similarity(
                        tfidf_matrix[idx:idx+1], 
                        tfidf_matrix[selected_idx:selected_idx+1]
                    )[0][0]
                    for selected_idx in selected_indices
                )
                if max_similarity > 0.7:  # 유사도가 높으면 건너뛰기
                    continue
            
            selected.append((idx, sent))
            selected_indices.add(idx)
        
        # 원래 순서대로 정렬
        selected.sort(key=lambda x: x[0])
        
        return [sent for _, sent in selected]