from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import io

app = FastAPI()
client = OpenAI()

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 또는 허용할 도메인을 명시적으로 설정합니다.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No selected file")
    
    try:
        # 파일 확장자와 MIME 타입 확인
        if file.content_type not in ['audio/flac', 'audio/m4a', 'audio/mp3', 'audio/mp4', 'audio/mpeg', 'audio/mpga', 'audio/oga', 'audio/ogg', 'audio/wav', 'audio/webm']:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        # 파일을 BytesIO로 변환
        file_data = await file.read()
        audio_bytes = io.BytesIO(file_data)
        audio_bytes.name = file.filename

        # OpenAI API로 파일 전송
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_bytes,
            response_format="text",
        )
        return JSONResponse(content={'transcription': transcription}, status_code=200)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
