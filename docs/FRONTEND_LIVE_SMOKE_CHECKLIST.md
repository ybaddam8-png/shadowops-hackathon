# Frontend Live Smoke Checklist

Follow these steps to manually verify the ShadowOps Judge UI integration:

## 1. Start Backend
Run the backend with FastAPI to serve the `/ready` endpoint and WebSocket connection.
```bash
cd backend-ml
python demo_judge.py 
# or uvicorn main:app
```

## 2. Start Frontend
Run the UI development server.
```bash
cd frontend
npm run dev
```

## 3. Check TopBar health/ready
- Open the UI in horizontal split mode.
- Verify the Topbar indicators for `Backend` health icon.
- It should ping `/health` or `/ready` and reflect the mode (e.g. `fallback` vs `model_ready`).

## 4. Click Decision Button
- Select an incident in the UI.
- Click `ALLOW`, `BLOCK`, `FORK` or `QUARANTINE`.
- Verify the action is pushed through the system correctly.

## 5. Verify LIVE BACKEND RESULT
- Ensure the result directly from the backend API reflects accurately in the incident timeline.
- Verify risk vector, mitigation plan, and missing evidence renders correctly.

## 6. Verify Fallback Label if Backend is Stopped
- Stop the `demo_judge.py` or FastAPI system.
- Perform another action in the UI.
- Verify that the UI correctly labels it as using the local isolated/fallback mode and does not crash out.

## 7. Check Browser Console
- Open browser developer tools (F12).
- Check the console for any unhandled Promise rejections, missing key properties, or WebSocket disconnect noise. All errors should be handled gracefully.
