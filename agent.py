"""
Lead Logger Voice Agent — agent.py
Full-duplex real-time voice agent using:
  Transport  : LiveKit (WebRTC)
  STT        : Sarvam AI (saaras:v2.5 streaming WebSocket)
  LLM        : OpenAI gpt-4o
  TTS        : Sarvam AI (SarvamTTSService WebSocket)
  Framework  : Pipecat (latest stable)

Run:
    python agent.py --room <LIVEKIT_ROOM_NAME>
"""

import asyncio
import os
import sys
import json
import argparse
import aiohttp

from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from loguru import logger

# ── Pipecat core ──────────────────────────────────────────────────────────────
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask

from pipecat.frames.frames import (
    EndFrame,
    LLMRunFrame,
    TTSSpeakFrame,
)

# ── Context / aggregators ─────────────────────────────────────────────────────
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)

# ── Tool / function calling ───────────────────────────────────────────────────
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams

# ── VAD ───────────────────────────────────────────────────────────────────────
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams

# ── LiveKit transport ─────────────────────────────────────────────────────────
from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport

# ── AI services ───────────────────────────────────────────────────────────────
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.sarvam.stt import SarvamSTTService, SarvamSTTSettings
from pipecat.services.sarvam.tts import SarvamTTSService, SarvamTTSSettings
from pipecat.transcriptions.language import Language

# ── LiveKit token generation ──────────────────────────────────────────────────
from livekit.api import AccessToken, VideoGrants

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SUBMIT_LEAD_URL = "https://cs9uag.buildship.run/submit-lead-interaction-log"
SESSION_TIMEOUT_SECS = 300         # 5-minute limit for the conversation
BOT_PARTICIPANT_NAME = "LeadLogger"

REQUIRED_FIELDS = [
    "interaction_status",
    "interaction_time",
    "interaction_type",
    "lead_source",
    "source_group",
    "lead_id",
    "remarks",
    "tenant_lead_status",
    "temperature",
    "cre_name",
]

OPTIONAL_FIELDS = ["agent_name", "outlet_name"]

# ─────────────────────────────────────────────────────────────────────────────
# LiveKit token helper
# ─────────────────────────────────────────────────────────────────────────────
def generate_livekit_token(room_name: str, participant_name: str) -> str:
    """Generate a LiveKit access token for the bot participant."""
    api_key    = os.environ["LIVEKIT_API_KEY"]
    api_secret = os.environ["LIVEKIT_API_SECRET"]

    token = (
        AccessToken(api_key, api_secret)
        .with_identity(participant_name)
        .with_name(participant_name)
        .with_grants(
            VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
            )
        )
        .to_jwt()
    )
    logger.info(f"Generated LiveKit token for room='{room_name}' identity='{participant_name}'")
    return token


# ─────────────────────────────────────────────────────────────────────────────
# submit_lead_details — the tool the LLM calls
# ─────────────────────────────────────────────────────────────────────────────
async def submit_lead_details(params: FunctionCallParams, **kwargs):
    """
    Submit collected lead interaction data to the CRM endpoint.

    Args:
        interaction_status: Current status of the interaction
        interaction_time: ISO 8601 datetime of the interaction
        interaction_type: Type/category of the interaction
        lead_source: Source where the lead originated
        source_group: Group classification of the lead source
        lead_id: Unique lead identifier
        remarks: Notes or comments about the interaction
        tenant_lead_status: Tenant-specific lead status
        temperature: Lead temperature — Hot, Warm, or Cold
        cre_name: Name of the CRE handling the lead
        agent_name: (optional) Agent name
        outlet_name: (optional) Outlet or branch name
    """
    args: dict = params.arguments  # FunctionCallParams bundles the args here

    # ── Validate all required fields are present ──────────────────────────────
    missing = [f for f in REQUIRED_FIELDS if not args.get(f)]
    if missing:
        logger.warning(f"submit_lead_details called with MISSING fields: {missing}")
        await params.result_callback(
            {"error": f"Missing required fields: {missing}. Do NOT call the API yet."}
        )
        return

    logger.info(f"✅ All required fields present. Submitting lead: {json.dumps(args, indent=2)}")

    # ── POST to CRM endpoint ──────────────────────────────────────────────────
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                SUBMIT_LEAD_URL,
                json=args,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                response_text = await resp.text()
                logger.info(
                    f"CRM API response — status={resp.status} body={response_text[:200]}"
                )
                result = {
                    "success": resp.status in (200, 201, 202),
                    "status_code": resp.status,
                    "body": response_text[:500],
                }
    except Exception as e:
        logger.error(f"CRM API request failed: {e}")
        result = {"success": False, "error": str(e)}

    await params.result_callback(result)


# ─────────────────────────────────────────────────────────────────────────────
# Tool schema
# ─────────────────────────────────────────────────────────────────────────────
lead_tool_schema = FunctionSchema(
    name="submit_lead_details",
    description=(
        "Submit a completed lead interaction log to the CRM. "
        "ONLY call this function when ALL 10 required fields have been collected "
        "from the user: interaction_status, interaction_time, interaction_type, "
        "lead_source, source_group, lead_id, remarks, tenant_lead_status, "
        "temperature, cre_name. Do NOT call if any required field is missing."
    ),
    properties={
        "interaction_status": {
            "type": "string",
            "description": "Current status of the interaction (e.g., Connected, Not Connected, Follow-up)",
        },
        "interaction_time": {
            "type": "string",
            "description": "ISO 8601 datetime string of when the interaction occurred",
        },
        "interaction_type": {
            "type": "string",
            "description": "Type of interaction (e.g., Call, WhatsApp, Email, Visit)",
        },
        "lead_source": {
            "type": "string",
            "description": "Source where the lead came from (e.g., 99acres, MagicBricks, Website)",
        },
        "source_group": {
            "type": "string",
            "description": "Group/category of the lead source (e.g., Digital, Referral, Walk-in)",
        },
        "lead_id": {
            "type": "string",
            "description": "Unique identifier for this lead",
        },
        "remarks": {
            "type": "string",
            "description": "Free-text remarks or notes about the interaction",
        },
        "tenant_lead_status": {
            "type": "string",
            "description": "Tenant-specific classification of this lead's status",
        },
        "temperature": {
            "type": "string",
            "enum": ["Hot", "Warm", "Cold"],
            "description": "Lead temperature: Hot, Warm, or Cold",
        },
        "cre_name": {
            "type": "string",
            "description": "Full name of the Customer Relationship Executive handling this lead",
        },
        "agent_name": {
            "type": "string",
            "description": "(Optional) Name of the sales agent",
        },
        "outlet_name": {
            "type": "string",
            "description": "(Optional) Name of the outlet or branch",
        },
    },
    required=REQUIRED_FIELDS,
)

tools = ToolsSchema(standard_tools=[lead_tool_schema])


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are a high-speed Lead Logger voice agent. You can interact in both English and Hindi. Your ONLY job is to collect
exactly 10 required fields from the CRE (Customer Relationship Executive) and then call
submit_lead_details.

REQUIRED FIELDS (collect ALL 10 before calling the tool):
1. interaction_status   – e.g., Connected / Not Connected / Follow-up
2. interaction_time     – when did it happen? Convert to ISO 8601 (use today {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} if "now")
3. interaction_type     – e.g., Call / WhatsApp / Email / Visit
4. lead_source          – e.g., 99acres / MagicBricks / Website / Walk-in
5. source_group         – e.g., Digital / Referral / Walk-in
6. lead_id              – the lead's unique ID or reference number
7. remarks              – brief notes about the interaction
8. tenant_lead_status   – tenant-specific status label
9. temperature          – MUST be exactly: Hot, Warm, or Cold
10. cre_name            – your (the CRE's) full name

OPTIONAL (collect if offered but do NOT block on them):
- agent_name
- outlet_name

RULES:
• Ask efficiently — combine 2-3 questions per turn when possible.
• If a field is ambiguous, clarify once and move on.
• NEVER call submit_lead_details until all 10 required fields are confirmed.
• After the tool returns success, say EXACTLY: "Logged successfully" — nothing else.
• Be efficient and fast, but ensure all fields are collected accurately.
• Speak concisely. No filler words. No pleasantries beyond the opening greeting.

Opening line: "Lead Logger ready. Give me your lead ID and CRE name to start."
"""


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline factory
# ─────────────────────────────────────────────────────────────────────────────
async def run_agent(room_name: str, aiohttp_session: aiohttp.ClientSession):
    livekit_url   = os.environ["LIVEKIT_URL"]
    token         = generate_livekit_token(room_name, BOT_PARTICIPANT_NAME)

    # ── LiveKit Transport ────────────────────────────────────────────────────
    transport = LiveKitTransport(
        url=livekit_url,
        token=token,
        room_name=room_name,
        params=LiveKitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
            audio_out_channels=1,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    stop_secs=0.5,          # slightly more relaxed end-of-turn
                    start_secs=0.2,         # ignore very short noises
                    confidence=0.7,         # higher confidence threshold
                )
            ),
        ),
    )

    # ── STT — Sarvam saaras:v3 streaming WebSocket ───────────────────────────
    stt = SarvamSTTService(
        api_key=os.environ["SARVAM_API_KEY"],
        settings=SarvamSTTSettings(
            model="saaras:v3",
            language=None,
        ),
    )

    # ── LLM — OpenAI gpt-4o ──────────────────────────────────────────────────
    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        model="gpt-4o",
    )

    # Register the tool handler (non-direct: LLM waits for result before continuing)
    llm.register_function("submit_lead_details", submit_lead_details)

    # ── TTS — Sarvam bulbul:v3 streaming WebSocket ───────────────────────────
    tts = SarvamTTSService(
        api_key=os.environ["SARVAM_API_KEY"],
        settings=SarvamTTSSettings(
            model="bulbul:v3",
            voice="aditya",
            language=Language.HI_IN,
            pace=1.3,               # slightly fast for the 30-second constraint
        ),
    )

    # ── LLM Context ──────────────────────────────────────────────────────────
    context = LLMContext(
        messages=[{"role": "system", "content": SYSTEM_PROMPT}],
        tools=tools,
    )
    context_aggregator = LLMContextAggregatorPair(context)

    # ── Pipeline: input → STT → LLM (with tools) → TTS → output ─────────────
    pipeline = Pipeline(
        [
            transport.input(),          # LiveKit mic audio in
            stt,                        # Sarvam STT → TranscriptionFrames
            context_aggregator.user(),  # accumulate user turns
            llm,                        # OpenAI gpt-4o (fires tool calls)
            tts,                        # Sarvam TTS → audio frames
            transport.output(),         # LiveKit speaker audio out
            context_aggregator.assistant(),  # accumulate assistant turns
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,   # full-duplex: user can interrupt bot
            enable_metrics=True,
        ),
    )

    # ── Event: first participant joins → greet and start LLM ─────────────────
    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant_id: str):
        logger.info(f"First participant joined: {participant_id}")
        # Wait 1.5s to ensure the user's browser has subscribed to the audio track
        await asyncio.sleep(1.5)
        await task.queue_frames([LLMRunFrame()])

    # ── Event: participant joins (fallback if first_participant event is missed) ──
    @transport.event_handler("on_participant_connected")
    async def on_participant_connected(transport, participant):
        logger.info(f"Participant connected: {participant.identity}")
        # If we haven't started yet, trigger greeting after a short sync delay
        await asyncio.sleep(1.0)
        await task.queue_frames([LLMRunFrame()])

    # ── Event: participant leaves → end session ───────────────────────────────
    @transport.event_handler("on_participant_disconnected")
    async def on_participant_disconnected(transport, participant_id: str):
        logger.info(f"Participant disconnected: {participant_id} — ending session")
        await task.queue_frames([EndFrame()])

    # ── Session timeout watchdog ──────────────────────────────────────────────
    async def session_timeout():
        await asyncio.sleep(SESSION_TIMEOUT_SECS)
        logger.warning(f"⏱  Session timeout ({SESSION_TIMEOUT_SECS}s) reached — terminating")
        await task.queue_frames([
            TTSSpeakFrame("Session timed out. Please try again."),
            EndFrame(),
        ])

    asyncio.create_task(session_timeout())

    # ── Run ───────────────────────────────────────────────────────────────────
    runner = PipelineRunner()
    await runner.run(task)
    logger.info("Pipeline finished.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser(description="Lead Logger Voice Agent")
    parser.add_argument(
        "--room",
        default=os.getenv("LIVEKIT_ROOM", "lead-logger-room"),
        help="LiveKit room name to join",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(
        sys.stdout,
        level="DEBUG",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        colorize=True,
    )

    logger.info(f"Starting Lead Logger agent in room: {args.room}")

    async with aiohttp.ClientSession() as session:
        await run_agent(args.room, session)


if __name__ == "__main__":
    asyncio.run(main())
