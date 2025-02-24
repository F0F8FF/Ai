import os
import requests
import tarfile
from tqdm import tqdm

def download_nuscenes():
    """nuScenes 데이터셋 다운로드 안내"""
    print("nuScenes 데이터셋 다운로드 방법:")
    print("\n1. https://www.nuscenes.org/nuscenes#download 에서 계정 생성")
    print("2. 로그인 후 'Download' 페이지 접속")
    print("3. 'v1.0-mini' 데이터셋 다운로드")
    print("4. 다운로드한 파일을 'data/nuscenes' 폴더에 압축 해제")
    
    # 데이터 저장 폴더 생성
    os.makedirs('data/nuscenes', exist_ok=True)
    
    print("\n데이터 저장 경로:")
    print(os.path.abspath('data/nuscenes'))
    
    print("\n다운로드 후 파일을 위 경로에 압축 해제해주세요.")

if __name__ == "__main__":
    download_nuscenes()