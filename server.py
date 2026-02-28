from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from modules.vae import VAE
from modules.rssm import RSSM
from modules.world_model import LFMWorldModel
from wrapper import Wrapper

def create_app(config):
    app = FastAPI(title="Cookie Run Game Server")
    
    # Static files
    static_path = "static"
    app.mount("/static", StaticFiles(directory=static_path), name="static")
    
    print("🔄 Loading resources...")
    vae = VAE(config)
    rssm = RSSM(config)
    world_model = LFMWorldModel(config, vae, rssm)
    print("✅ Resources loaded.")
    
    @app.get("/", response_class=HTMLResponse)
    async def read_root(): 
        html_file = f"{static_path}/index.html"
        with open(html_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()

        wrapper = Wrapper(
            config=config,
            world_model=world_model,
        )
        
        try:
            img = wrapper.reset()
            img_base64 = wrapper.image_to_base64(img)

            await websocket.send_json({
                "status": "success",
                "image": img_base64,
                "current_action": "reset"
            })

            while True:
                data = await websocket.receive_json()
                action_type = data.get("type")
                action = data.get("action", "none")

                if action_type == "reset":
                    img = wrapper.reset()
                elif action_type == "action":
                    img = wrapper.step(action)
                else:
                    continue
                
                img_base64 = wrapper.image_to_base64(img)

                await websocket.send_json({
                    "status": "success",
                    "image": img_base64,
                    "current_action": action
                })
                
        except WebSocketDisconnect:
            print(f"Client disconnected")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"WebSocket error: {e}")
            try:
                await websocket.send_json({
                    "status": "error",
                    "message": str(e)
                })
            except:
                pass
    return app