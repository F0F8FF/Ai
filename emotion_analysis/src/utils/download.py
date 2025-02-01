import os
import requests
import zipfile
from tqdm import tqdm

def download_ravdess():
    # RAVDESS 데이터셋 URL
    url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
    
    # 저장 경로 설정
    data_dir = "data/raw/ravdess"
    os.makedirs(data_dir, exist_ok=True)
    
    # 파일 다운로드
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    zip_path = os.path.join(data_dir, "ravdess.zip")
    
    print("RAVDESS 데이터셋 다운로드 중...")
    with open(zip_path, 'wb') as file, tqdm(
        desc=zip_path,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)
    
    # 압축 해제
    print("압축 해제 중...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    # 압축 파일 삭제
    os.remove(zip_path)
    print("다운로드 완료!")

if __name__ == "__main__":
    download_ravdess() 