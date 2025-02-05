# Korean News Summarizer

A powerful news summarization system that provides both extractive (TextRank) and abstractive (KoBART) summarization for Korean news articles.

## Features
- News article summarization via URL or direct text input
- Extractive summarization using TextRank algorithm
- Abstractive summarization using KoBART model
- Web interface for easy access
- Automatic metadata removal
- Duplicate content filtering
- Expert quote prioritization


## Technical Details

### Extractive Summarization
- TextRank algorithm implementation
- TF-IDF vectorization
- Cosine similarity for sentence relationships
- Custom weighting system:
  - Keyword importance
  - Expert quotes
  - Sentence position
  - Length normalization

### Abstractive Summarization
- Based on KoBART model
- Transformer architecture
- Beam search decoding
- Length-controlled generation

## Requirements
- Python 3.8 or higher
- PyTorch
- Transformers
- MeCab
- Flask
- NetworkX
- Scikit-learn
- Newspaper3k

## Performance Optimization
- Intelligent sentence selection
- Duplicate content detection
- Quote preservation
- Metadata filtering
- Position-based weighting

## Known Limitations
- Requires significant RAM (minimum 4GB)
- GPU recommended for abstractive summarization
- Limited to Korean language content
- URL access may be restricted for some news sites


## License
MIT License

## Acknowledgments
- KoBART model from SKT
- TextRank algorithm implementation
- Korean NLP community
