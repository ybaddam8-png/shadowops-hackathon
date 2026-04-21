import asyncio
import websockets

async def listen():
    uri = "ws://localhost:8000/ws/telemetry"
    print(f"Connecting to {uri}...")
    try:
        # INJECT THE ORIGIN HEADER HERE
        headers = {"Origin": "http://localhost:8000"}
        async with websockets.connect(uri, extra_headers=headers) as websocket:
            print("✅ Connected! Waiting for ShadowOps telemetry...\n")
            while True:
                message = await websocket.recv()
                print(f"Payload Received:\n{message}\n{'-'*40}")
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(listen())