import io
from openai import OpenAI
from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
    UploadFile,
    File,
    HTTPException,
    Request,
)
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

router = APIRouter()


# @router.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     try:
#         while True:
#             data = await websocket.receive_bytes()
#             audio_bytes = io.BytesIO(data)
#             audio_bytes.name = "audio.wav"

#             # STT
#             openai_client = websocket.app.state.openai_client
#             transcription = openai_client.audio.transcriptions.create(
#                 model="whisper-1",
#                 file=audio_bytes,
#                 response_format="text",
#             )

#             # Model answer
#             transcription = "대한민국의 수도는 뭔지 간단하게 설명해줘"
#             response = openai_client.chat.completions.create(
#                 model="gpt-3.5-turbo",
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": "주어진 질문에 대해서 간단하고 친절하게 대답해주세요. 반드시 한국어로 대답해주세요.",
#                     },
#                     {"role": "user", "content": transcription},
#                 ],
#                 temperature=0.7,
#             )
#             model_answer = response.choices[0].message.content

#             # TTS
#             tts_response = openai_client.audio.speech.create(
#                 model="tts-1",
#                 voice="alloy",
#                 input=model_answer,
#             )

#             # Stream the audio data to the client
#             chunk_size = 4096
#             for chunk in tts_response.iter_bytes(chunk_size=chunk_size):
#                 await websocket.send_bytes(chunk)

#     except WebSocketDisconnect:
#         raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/tts")
async def tts_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            openai_client = websocket.app.state.openai_client

            # TTS
            tts_response = openai_client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=data,
            )

            # Stream the audio data to the client
            chunk_size = 4096
            for chunk in tts_response.iter_bytes(chunk_size=chunk_size):
                await websocket.send_bytes(chunk)

    except Exception as e:
        print(f"Error in WebSocket: {str(e)}")
    finally:
        await websocket.close()


from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import asyncio
from typing import List
import logging
from pydantic import BaseModel


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic model for error responses
class ErrorResponse(BaseModel):
    error: str


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)


manager = ConnectionManager()


@router.websocket("/tts")
async def tts_websocket(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()

            # TTS processing
            try:
                tts_response = await asyncio.to_thread(
                    websocket.app.openai_client.audio.speech.create,
                    model="tts-1",
                    voice="alloy",
                    input=data,
                )

                # Stream the audio data to the client
                chunk_size = 1024  # Adjust this based on your needs
                for chunk in tts_response.iter_bytes(chunk_size=chunk_size):
                    await websocket.send_bytes(chunk)

                # Signal end of audio stream
                await websocket.send_json({"status": "completed"})

            except Exception as e:
                logger.error(f"Error in TTS processing: {str(e)}")
                await websocket.send_json(
                    ErrorResponse(error=f"TTS processing failed: {str(e)}").dict()
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket: {str(e)}")
        await websocket.send_json(
            ErrorResponse(error=f"Unexpected error: {str(e)}").dict()
        )
    finally:
        manager.disconnect(websocket)
