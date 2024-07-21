from fastapi import APIRouter, File, Form, UploadFile, Request, HTTPException
import io
import openai

router = APIRouter()


@router.post("/stt")
async def transcribe_audio(
    request: Request,
    file: UploadFile = File(...),
    user_id: str = Form(...),
    request_id: str = Form(...),
    timestamp: str = Form(...),
):
    openai_client = request.app.state.openai_client
    if not file.filename:
        raise HTTPException(status_code=400, detail="No selected file")
    try:
        contents = await file.read()
        audio_bytes = io.BytesIO(contents)
        audio_bytes.name = file.filename

        # STT
        transcription = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_bytes,
            response_format="text",
        )

        # Model answer
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 친절하고 전문적인 아동 심리 상담사입니다. 아동이 하는 말에 따라 친절하게 응답해주세요. 반드시 두 문장 이내로 짧게 대답해주세요.",
                },
                {"role": "user", "content": transcription},
            ],
            temperature=0.7,
        )
        model_answer = response.choices[0].message.content

        return {
            "user_id": user_id,
            "request_id": request_id,
            "timestamp": timestamp,
            "transcription": transcription,
            "model_answer": model_answer,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
