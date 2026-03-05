import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from modules.vqvae import VQVAE
from modules.rssm import RSSM
from wrapper import Wrapper

def create_app(config):
    app = FastAPI(title="Cookie Run Game Server")
    
    # Static files
    static_path = "static"
    app.mount("/static", StaticFiles(directory=static_path), name="static")
    
    print("🔄 Loading resources...")
    vqvae = VQVAE(config).to(config.device)
    vqvae.load_vqvae(config.vqvae_path)
    codebook_weight = vqvae.quantizer.embedding.clone().detach()

    rssm = RSSM(config, codebook_weight).to(config.device)
    rssm.load_rssm(config.rssm_path)
    print("✅ Resources loaded.")
    
    @app.get("/", response_class=HTMLResponse)
    async def read_root(): 
        html_file = f"{static_path}/index.html"
        with open(html_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        wrapper = Wrapper(config=config, vqvae=vqvae, rssm=rssm)
        
        latest_action = "none" 
        running = True
        
        async def receive_actions():
            nonlocal latest_action, running
            try:
                while running:
                    data = await websocket.receive_json()
                    if data.get("type") == "reset":
                        latest_action = "reset"
                    else:
                        latest_action = data.get("action", "none")
            except WebSocketDisconnect:
                running = False
            except Exception:
                running = False
        
        async def generate_frames():
            nonlocal latest_action, running
            TARGET_FPS = 20
            FRAME_INTERVAL = 1.0 / TARGET_FPS  # 50ms
            
            img = await asyncio.to_thread(wrapper.reset)
            img_base64 = await asyncio.to_thread(wrapper.image_to_base64, img)
            await websocket.send_json({
                "status": "success",
                "image": img_base64,
                "current_action": "reset"
            })
            
            while running:
                frame_start = asyncio.get_event_loop().time()
                
                action = latest_action
                if action in ("jump", "reset"):
                    latest_action = "none"
            
                try:
                    if action == "reset":
                        img = await asyncio.to_thread(wrapper.reset)
                    else:
                        img = await asyncio.to_thread(wrapper.step, action)
                    
                    img_base64 = await asyncio.to_thread(
                        wrapper.image_to_base64, img
                    )
                    
                    await websocket.send_json({
                        "status": "success",
                        "image": img_base64,
                        "current_action": action
                    })
                except WebSocketDisconnect:
                    running = False
                    break
                except Exception as e:
                    print(f"Frame error: {e}")
                    running = False
                    break
                
                elapsed = asyncio.get_event_loop().time() - frame_start
                sleep_time = FRAME_INTERVAL - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
        try:
            await asyncio.gather(
                receive_actions(),
                generate_frames()
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"WebSocket error: {e}")

    return app