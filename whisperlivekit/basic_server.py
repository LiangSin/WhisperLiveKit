from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from whisperlivekit import TranscriptionEngine, AudioProcessor, get_inline_ui_html, parse_args
from whisperlivekit.ip_middleware import create_ip_middleware
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

args = parse_args()
transcription_engine = None


def _build_session_dir_name(archive_root: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_path = Path(archive_root)
    candidate = timestamp
    suffix = 1
    while (base_path / candidate).exists():
        candidate = f"{timestamp}_{suffix:02d}"
        suffix += 1
    return candidate

@asynccontextmanager
async def lifespan(app: FastAPI):    
    global transcription_engine
    transcription_engine = TranscriptionEngine(
        **vars(args),
    )
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize IP restriction checker if configured
ip_checker = None
if args.allowed_ips or args.allowed_networks:
    allowed_ips = args.allowed_ips.split(',') if args.allowed_ips else None
    allowed_networks = args.allowed_networks.split(',') if args.allowed_networks else None

    ip_checker = create_ip_middleware(
        allowed_ips=allowed_ips,
        allowed_networks=allowed_networks
    )
    app.middleware("http")(ip_checker)

@app.get("/")
async def get():
    return HTMLResponse(get_inline_ui_html())

@app.get("/health")
async def health():
    return {"status": "ok"}

async def handle_websocket_results(websocket, results_generator):
    """Consumes results from the audio processor and sends them via WebSocket."""
    try:
        async for response in results_generator:
            await websocket.send_json(response.to_dict())
        # when the results_generator finishes it means all audio has been processed
        logger.info("Results generator finished. Sending 'ready_to_stop' to client.")
        await websocket.send_json({"type": "ready_to_stop"})
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected while handling results (client likely closed connection).")
    except Exception as e:
        logger.exception(f"Error in WebSocket results handler: {e}")


@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    # Check IP restrictions for WebSocket connections
    if ip_checker:
        client_ip = None
        # Check X-Forwarded-For header (for proxies)
        x_forwarded_for = websocket.headers.get("X-Forwarded-For")
        if x_forwarded_for:
            client_ip = x_forwarded_for.split(",")[0].strip()
        else:
            # Direct connection - use websocket.client.host
            client_ip = websocket.client.host if websocket.client else None

        if not client_ip:
            logger.warning("Could not determine client IP address for WebSocket")
            await websocket.close(code=1008)  # Policy violation
            return

        if not ip_checker.is_ip_allowed(client_ip):
            logger.warning(f"WebSocket access denied for IP: {client_ip}")
            await websocket.close(code=1008)  # Policy violation
            return

    global transcription_engine
    audio_processor_kwargs = {"transcription_engine": transcription_engine}
    if getattr(args, "archive_enabled", False):
        audio_processor_kwargs["session_dir_name"] = _build_session_dir_name(args.archive_dir)

    await websocket.accept()
    logger.info("WebSocket connection opened.")

    try:
        await websocket.send_json({"type": "config", "useAudioWorklet": bool(args.pcm_input)})
    except Exception as e:
        logger.warning(f"Failed to send config to client: {e}")

    # The AudioProcessor is created lazily on the first audio chunk so that
    # an optional client config message (a JSON text frame sent before any
    # audio) can customize the session, e.g. per-connection prompts.
    audio_processor = None
    websocket_task = None

    async def handle_client_config(text):
        try:
            control = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Ignoring non-JSON text frame from client.")
            return
        if not isinstance(control, dict) or control.get("type") != "config":
            logger.warning(f"Ignoring unknown client message: {control!r}")
            return
        if audio_processor is not None:
            logger.warning("Client config received after audio started; ignoring.")
            await websocket.send_json({
                "type": "config_error",
                "message": "config must be sent before the first audio chunk",
            })
            return
        applied = {}
        for key in ("init_prompt", "static_init_prompt"):
            if key in control and control[key] is not None:
                value = control[key]
                if not isinstance(value, str):
                    await websocket.send_json({
                        "type": "config_error",
                        "message": f"{key} must be a string",
                    })
                    return
                audio_processor_kwargs[key] = value
                applied[key] = value
        await websocket.send_json({"type": "config_ack", "applied": applied})
        logger.info(f"Applied client session config: {applied}")

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(message.get("code") or 1000)
            if message.get("text") is not None:
                await handle_client_config(message["text"])
                continue
            data = message.get("bytes")
            if data is None:
                continue
            if audio_processor is None:
                audio_processor = AudioProcessor(**audio_processor_kwargs)
                results_generator = await audio_processor.create_tasks()
                websocket_task = asyncio.create_task(
                    handle_websocket_results(websocket, results_generator)
                )
            await audio_processor.process_audio(data)
    except KeyError as e:
        if 'bytes' in str(e):
            logger.warning(f"Client has closed the connection.")
        else:
            logger.error(f"Unexpected KeyError in websocket_endpoint: {e}", exc_info=True)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client during message receiving loop.")
    except Exception as e:
        logger.error(f"Unexpected error in websocket_endpoint main loop: {e}", exc_info=True)
    finally:
        logger.info("Cleaning up WebSocket endpoint...")
        if websocket_task is not None:
            if not websocket_task.done():
                websocket_task.cancel()
            try:
                await websocket_task
            except asyncio.CancelledError:
                logger.info("WebSocket results handler task was cancelled.")
            except Exception as e:
                logger.warning(f"Exception while awaiting websocket_task completion: {e}")

        if audio_processor is not None:
            await audio_processor.cleanup()
        logger.info("WebSocket endpoint cleaned up successfully.")

def main():
    """Entry point for the CLI command."""
    import uvicorn
    
    uvicorn_kwargs = {
        "app": "whisperlivekit.basic_server:app",
        "host":args.host, 
        "port":args.port, 
        "reload": False,
        "log_level": "info",
        "lifespan": "on",
    }
    
    ssl_kwargs = {}
    if args.ssl_certfile or args.ssl_keyfile:
        if not (args.ssl_certfile and args.ssl_keyfile):
            raise ValueError("Both --ssl-certfile and --ssl-keyfile must be specified together.")
        ssl_kwargs = {
            "ssl_certfile": args.ssl_certfile,
            "ssl_keyfile": args.ssl_keyfile
        }

    if ssl_kwargs:
        uvicorn_kwargs = {**uvicorn_kwargs, **ssl_kwargs}
    if args.forwarded_allow_ips:
        uvicorn_kwargs = { **uvicorn_kwargs, "forwarded_allow_ips" : args.forwarded_allow_ips }

    uvicorn.run(**uvicorn_kwargs)

if __name__ == "__main__":
    main()
