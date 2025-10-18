import numpy as np
import cv2
import json
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from detection_webs import process_frame_and_get_status, reset_all_states

app = FastAPI()

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/capture")
async def capture_ws(ws: WebSocket):
    await ws.accept()
    reset_all_states()

    try:
        while True:
            # Receive message
            message = await ws.receive_text()

            try:
                data = json.loads(message)

                if data.get("type") == "ping":
                    await ws.send_text(json.dumps({
                        "type": "pong",
                        "status": "Server alive"
                    }))
                    continue

                elif data.get("type") == "reset":
                    reset_all_states()
                    await ws.send_text(json.dumps({
                        "status": "🔄 Detection reset",
                        "color": "blue"
                    }))
                    continue

                elif data.get("type") == "frame":
                    frame_data = data.get("data", "")
                    if not frame_data:
                        await ws.send_text(json.dumps({
                            "status": "❌ No frame data received",
                            "color": "red"
                        }))
                        continue

                    try:
                        # Strip base64 prefix if present
                        if frame_data.startswith("data:image"):
                            frame_data = frame_data.split(",")[1]

                        # Decode frame
                        nparr = np.frombuffer(base64.b64decode(frame_data), np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                        if frame is None:
                            await ws.send_text(json.dumps({
                                "status": "❌ Invalid frame data",
                                "color": "red"
                            }))
                            continue

                        # Process frame
                        result = process_frame_and_get_status(frame)

                        # Prepare response
                        payload = {
                            "status": result["status"],
                            "color": result["color"],
                            "countdown": result.get("countdown"),
                            "captured": result.get("captured", False),
                            "image": None,
                            "filename": None,
                            "scan_step": result.get("scan_step"),
                            "scan_progress": result.get("scan_progress"),
                            "scan_completed": result.get("scan_completed", False),
                        }

                        # Optional: include debug information for 3D scan
                        if "obstruction_debug" in result:
                            payload["debug"] = result["obstruction_debug"]
                        if "turn_debug" in result:
                            payload["turn_debug"] = result["turn_debug"]
                        if "scan_info" in result:
                            payload["scan_info"] = result["scan_info"]
                        if "profile_snapshots" in result:
                            payload["profile_snapshots"] = result["profile_snapshots"]
                        if "depth_info" in result:
                            payload["depth_info"] = result["depth_info"]

                        # If photo captured, send final image
                        if result.get("captured") and "final_crop" in result:
                            success, buffer = cv2.imencode('.jpg', result["final_crop"])
                            if success:
                                image_base64 = base64.b64encode(buffer).decode('utf-8')
                                payload["image"] = image_base64
                                # Use 3D verified filename if scan was completed
                                if result.get("scan_completed"):
                                    payload["filename"] = "passport_photo_3d_verified.jpg"
                                else:
                                    payload["filename"] = "passport_photo.jpg"

                            if result.get("should_reset"):
                                reset_all_states()

                        await ws.send_text(json.dumps(payload))

                    except Exception as decode_error:
                        print(f"Frame decode error: {decode_error}")
                        await ws.send_text(json.dumps({
                            "status": "❌ Frame processing error",
                            "color": "red"
                        }))
                        continue

                else:
                    await ws.send_text(json.dumps({
                        "status": "❌ Unknown message type",
                        "color": "red"
                    }))

            except json.JSONDecodeError:
                await ws.send_text(json.dumps({
                    "status": "❌ Invalid JSON message",
                    "color": "red"
                }))
                continue

    except WebSocketDisconnect:
        print("WebSocket disconnected")

    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await ws.send_text(json.dumps({
                "status": "❌ Server error occurred",
                "color": "red"
            }))
        except:
            pass
        await ws.close()

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🚀 Starting 3D Passport Capture Server...")
    print("=" * 60)
    print("📡 WebSocket endpoint: ws://localhost:8765/ws/capture")
    print("🌐 Frontend should connect to this endpoint!")
    print("🎯 3D Face Scan Verification: ENABLED")
    print("✅ Features: Head Turn Detection | Depth Analysis | Liveness Check")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8765)

