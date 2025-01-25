import speech_recognition as sr
import datetime
import webbrowser
import os
import random

def listen_and_recognize():
    recognizer = sr.Recognizer()
    
    # 인사말 목록
    greetings = [
        "안녕하세요!",
        "반갑습니다!",
        "무엇을 도와드릴까요?",
        "말씀해주세요!"
    ]
    
    # 명령어 도움말
    commands = {
        "안녕": "인사에 답합니다",
        "시간": "현재 시간을 알려줍니다",
        "날짜": "오늘 날짜를 알려줍니다",
        "검색": "'검색 [검색어]' 형식으로 말하면 웹 검색을 합니다",
        "음악": "음악 재생 (음악 폴더 필요)",
        "메모": "'메모 [내용]' 형식으로 말하면 메모를 저장합니다",
        "도움말": "사용 가능한 명령어를 보여줍니다",
        "종료": "프로그램을 종료합니다"
    }
    
    print("음성 인식 시작... (Ctrl+C를 누르면 종료)")
    print(f"시작 인사: {random.choice(greetings)}")
    print("\n사용 가능한 명령어를 보려면 '도움말'이라고 말씀해주세요.")
    
    while True:
        try:
            with sr.Microphone() as source:
                # 주변 소음 조정
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # 음성 듣기
                print("\n듣고 있습니다...")
                audio = recognizer.listen(source)
                
                try:
                    # 음성을 텍스트로 변환
                    text = recognizer.recognize_google(audio, language='ko-KR')
                    print(f"\n인식된 텍스트: {text}")
                    
                    # 명령어 처리
                    if "안녕" in text:
                        print(f"응답: {random.choice(greetings)}")
                    
                    elif "시간" in text:
                        current_time = datetime.datetime.now().strftime("%H시 %M분")
                        print(f"현재 시각은 {current_time}입니다.")
                    
                    elif "날짜" in text:
                        current_date = datetime.datetime.now().strftime("%Y년 %m월 %d일")
                        print(f"오늘은 {current_date}입니다.")
                    
                    elif text.startswith("검색"):
                        search_query = text.replace("검색", "").strip()
                        if search_query:
                            url = f"https://www.google.com/search?q={search_query}"
                            webbrowser.open(url)
                            print(f"'{search_query}' 검색 중...")
                        else:
                            print("검색어를 말씀해주세요.")
                    
                    elif text.startswith("메모"):
                        memo_content = text.replace("메모", "").strip()
                        if memo_content:
                            with open("voice_memos.txt", "a", encoding="utf-8") as f:
                                f.write(f"{datetime.datetime.now()}: {memo_content}\n")
                            print("메모가 저장되었습니다.")
                        else:
                            print("메모할 내용을 말씀해주세요.")
                    
                    elif "도움말" in text:
                        print("\n=== 사용 가능한 명령어 ===")
                        for cmd, desc in commands.items():
                            print(f"- {cmd}: {desc}")
                    
                    elif "종료" in text:
                        print("프로그램을 종료합니다.")
                        break
                    
                except sr.UnknownValueError:
                    print("음성을 인식하지 못했습니다.")
                except sr.RequestError as e:
                    print(f"음성 인식 서비스 에러: {e}")
                
        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")
            break

if __name__ == "__main__":
    listen_and_recognize()
