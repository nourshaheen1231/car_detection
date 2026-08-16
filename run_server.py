import uvicorn
from bridge.server import app

if __name__ == "__main__":
    import os
    port = int(os.getenv("AI_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)