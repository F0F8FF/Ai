import speech_recognition as sr
import datetime
import openai
import os

# OpenAI API 키 설정
openai.api_key = 'your-api-key-here'

def get_ai_response(text):
    try:
        # GPT에게 응답 요청
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 친절한 한국어 음성 비서입니다. 자연스럽고 간단한 답변을 제공해주세요."},
                {"role": "user", "content": text}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API 에러: {e}")
        return "죄송합니다. 일시적인 오류가 발생했습니다."

def listen_and_recognize():
    recognizer = sr.Recognizer()
    
    print("음성 인식 시작... (Ctrl+C를 누르면 종료)")
    print("말씀해주세요...")
    
    while True:
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                print("\n듣고 있습니다...")
                audio = recognizer.listen(source)
                
                try:
                    text = recognizer.recognize_google(audio, language='ko-KR')
                    print(f"\n인식된 텍스트: {text}")
                    
                    if "종료" in text:
                        print("프로그램을 종료합니다.")
                        break
                    else:
                        # AI 응답 생성
                        response = get_ai_response(text)
                        print(f"AI 응답: {response}")
                    
                except sr.UnknownValueError:
                    print("음성을 인식하지 못했습니다.")
                except sr.RequestError as e:
                    print(f"음성 인식 서비스 에러: {e}")
                
        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")
            break

if __name__ == "__main__":
    listen_and_recognize()
