import io
from openai import OpenAI
from fastapi.responses import JSONResponse
from fastapi import APIRouter, HTTPException, Request, UploadFile, File
from pathlib import Path

router = APIRouter()


@router.post("/transcribe")
async def stt_audio(request: Request, file: UploadFile = File(...)):
    openai_client = request.app.state.openai_client
    if not file.filename:
        raise HTTPException(status_code=400, detail="No selected file")

    try:
        # 파일 확장자와 MIME 타입 확인
        if file.content_type not in [
            "audio/flac",
            "audio/m4a",
            "audio/mp3",
            "audio/mp4",
            "audio/mpeg",
            "audio/mpga",
            "audio/oga",
            "audio/ogg",
            "audio/wav",
            "audio/webm",
        ]:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        # 파일을 BytesIO로 변환
        file_data = await file.read()

        # root_dir = Path(__file__).resolve().parents[4]
        # local_file_path = root_dir / "data" / "test_sound.wav"
        # with open(local_file_path, "rb") as audio:
        #     file_data = audio.read()

        audio_bytes = io.BytesIO(file_data)
        audio_bytes.name = file.filename

        # 파일 포인터를 처음으로 이동
        audio_bytes.seek(0)

        # OpenAI API로 파일 전송
        transcription = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_bytes,
            response_format="text",
        )
        return JSONResponse(content={"transcription": transcription}, status_code=200)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
