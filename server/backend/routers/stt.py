import io
from openai import OpenAI
from fastapi.responses import JSONResponse
from fastapi import APIRouter, HTTPException, Request, UploadFile, File
from pathlib import Path

router = APIRouter()


@router.post("/stt")
async def transcribe_audio(request: Request, file: UploadFile = File(...)):
    openai_client = request.app.state.openai_client
    if not file.filename:
        raise HTTPException(status_code=400, detail="No selected file")
    try:
        # # 파일 확장자와 MIME 타입 확인
        # if file.content_type not in [
        #     "audio/flac",
        #     "audio/m4a",
        #     "audio/mp3",
        #     "audio/mp4",
        #     "audio/mpeg",
        #     "audio/mpga",
        #     "audio/oga",
        #     "audio/ogg",
        #     "audio/wav",
        #     "audio/webm",
        # ]:
        #     raise HTTPException(status_code=400, detail="Unsupported file format")

        contents = await file.read()

        # root_dir = Path(__file__).resolve().parents[4]
        # local_file_path = root_dir / "data" / "test_sound.wav"
        # with open(local_file_path, "rb") as audio:
        #     file_data = audio.read()

        audio_bytes = io.BytesIO(contents)
        audio_bytes.name = file.filename

        # STT
        transcription = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_bytes,
            response_format="text",
        )

        # Model answer
        transcription = "대한민국의 여름을 한 문장으로 표현하면?"
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "주어진 질문에 대해서 간단하고 친절하게 대답해주세요. 반드시 한국어로 대답해주세요.",
                },
                {"role": "user", "content": transcription},
            ],
            temperature=0.7,
        )
        model_answer = response.choices[0].message.content

        return {"transcription": transcription, "model_answer": model_answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
