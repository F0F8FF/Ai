from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration, AutoConfig
import torch

class KoBART_Summarizer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_name = 'gogamza/kobart-summarization'
        
        # 설정 먼저 로드
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = 3  # 명시적으로 설정
        
        # 토크나이저 초기화
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            model_name,
            bos_token='</s>',
            eos_token='</s>',
            unk_token='<unk>',
            pad_token='<pad>',
            mask_token='<mask>'
        )
        
        # 모델 초기화
        self.model = BartForConditionalGeneration.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True
        ).to(self.device)
        
        # 모델을 평가 모드로 설정
        self.model.eval()
    
    def summarize(self, text, max_length=128):
        try:
            with torch.no_grad():  # 추론 시 그래디언트 계산 비활성화
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    max_length=1024,
                    truncation=True,
                    padding='max_length'
                )
                
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                summary_ids = self.model.generate(
                    inputs['input_ids'],
                    max_length=max_length,
                    min_length=32,
                    length_penalty=1.0,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                    use_cache=True
                )
                
                return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                
        except Exception as e:
            print(f"Error in abstractive summarization: {str(e)}")
            return None