import os
import asyncio
from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from livekit.api import AccessToken, VideoGrants
from dotenv import load_dotenv
from loguru import logger
import aiohttp

# Import the agent logic
from agent import run_agent, BOT_PARTICIPANT_NAME

load_dotenv()

app = FastAPI(title="Lead Logger API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Keep track of active rooms to avoid spawning multiple agents for the same room
active_agents = set()

@app.get("/")
async def health_check():
    return {"status": "online", "message": "Lead Logger API is running"}

@app.get("/token")
async def get_token(
    room: str = Query(..., description="LiveKit room name"),
    identity: str = Query(..., description="Participant identity"),
    background_tasks: BackgroundTasks = None
):
    api_key    = os.environ.get("LIVEKIT_API_KEY")
    api_secret = os.environ.get("LIVEKIT_API_SECRET")

    if not api_key or not api_secret:
        raise HTTPException(status_code=500, detail="LiveKit credentials not configured")

    # 1. Generate Token for the Bubble User
    token = (
        AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(
            VideoGrants(
                room_join=True,
                room=room,
                can_publish=True,
                can_subscribe=True,
            )
        )
        .to_jwt()
    )

    # 2. Automatically start the AI Agent if it's not already in the room
    if room not in active_agents:
        logger.info(f"🚀 Starting AI agent for room: {room}")
        background_tasks.add_task(launch_agent, room)

    return {"token": token}

async def launch_agent(room_name: str):
    """Background task to run the Pipecat agent."""
    active_agents.add(room_name)
    try:
        async with aiohttp.ClientSession() as session:
            await run_agent(room_name, session)
    except Exception as e:
        logger.error(f"❌ Agent failed in room {room_name}: {e}")
    finally:
        active_agents.discard(room_name)
        logger.info(f"👋 Agent left room: {room_name}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
