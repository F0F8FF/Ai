import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import copy

# 이미지 로드 및 전처리 함수
def load_image(image_path, size=512):
    image = Image.open(image_path)
    loader = transforms.Compose([
        transforms.Resize([size, size]),  # 이미지 크기 조정
        transforms.ToTensor()  # 텐서로 변환
    ])
    image = loader(image).unsqueeze(0)
    return image

# VGG19 모델의 특정 레이어에서 특징 추출
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.chosen_features = ['0', '5', '10', '19', '28']  # 사용할 레이어
        self.model = models.vgg19(pretrained=True).features[:29]
    
    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features

# 스타일 손실 계산
def calc_style_loss(gen_features, style_features):
    style_loss = 0
    for gen_feature, style_feature in zip(gen_features, style_features):
        batch_size, channel, height, width = gen_feature.shape
        G = torch.mm(gen_feature.view(channel, height * width), 
                    gen_feature.view(channel, height * width).t())
        A = torch.mm(style_feature.view(channel, height * width),
                    style_feature.view(channel, height * width).t())
        style_loss += torch.mean((G - A) ** 2)
    return style_loss

# 콘텐츠 손실 계산
def calc_content_loss(gen_features, content_features):
    content_loss = torch.mean((gen_features[2] - content_features[2]) ** 2)
    return content_loss

def style_transfer(content_path, style_path, num_steps=500):
    # 이미지 로드
    content_img = load_image(content_path)
    style_img = load_image(style_path)
    
    # 생성할 이미지 초기화 (content 이미지로 시작)
    generated_img = content_img.clone().requires_grad_(True)
    
    # 모델 초기화
    model = VGG()
    
    # 최적화 도구 설정 (학습률 조정)
    optimizer = optim.Adam([generated_img], lr=0.001)
    
    print("스타일 변환 시작...")
    for step in range(num_steps):
        # 특징 추출
        gen_features = model(generated_img)
        content_features = model(content_img)
        style_features = model(style_img)
        
        # 손실 계산 (스타일 가중치 증가)
        style_loss = calc_style_loss(gen_features, style_features)
        content_loss = calc_content_loss(gen_features, content_features)
        total_loss = content_loss + style_loss * 150  # 스타일 가중치 증가
        
        # 역전파
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            print(f'Step {step}: Style Loss: {style_loss.item():.4f}, Content Loss: {content_loss.item():.4f}')
    
    print("변환 완료!")
    return generated_img

# 이미지 저장 함수
def save_image(tensor, filename):
    image = tensor.clone().squeeze(0)
    image = transforms.ToPILImage()(image)
    image.save(filename)

# 메인 실행
if __name__ == "__main__":
    content_path = "content.jpg"  # 변환할 이미지
    style_path = "style.jpg"    # 스타일 이미지
    
    # 스타일 변환 실행
    generated_img = style_transfer(content_path, style_path)
    
    # 결과 저장
    save_image(generated_img, "generated.jpg")
