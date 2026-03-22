from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from modules.rssm import RSSM
from modules.vqvae import VQVAE
from wrapper import Wrapper


def create_app(config):
    app = FastAPI(title="Cookie Run Game Server")

    static_path = "static"
    app.mount("/static", StaticFiles(directory=static_path), name="static")

    print("🔄 Loading resources...")
    vqvae = VQVAE(config).to(config.device)
    vqvae.load_vqvae(config.vqvae_path)
    vqvae.change_train_mode(train=False)

    codebook_weight = vqvae.quantizer.codebook.clone().detach()

    rssm = RSSM(config, codebook_weight=codebook_weight).to(config.device)
    rssm.load_rssm(config.rssm_path)
    rssm.change_train_mode(train=False)
    print("✅ Resources loaded.")

    @app.get("/", response_class=HTMLResponse)
    async def read_root():
        html_file = f"{static_path}/index.html"
        with open(html_file, "r", encoding="utf-8") as f:
            return f.read()

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()

        wrapper = Wrapper(
            config=config,
            vqvae=vqvae,
            rssm=rssm,
        )

        try:
            img = wrapper.reset()
            img_base64 = wrapper.image_to_base64(img)

            await websocket.send_json({
                "status": "success",
                "image": img_base64,
                "current_action": "reset",
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
                    "current_action": action,
                })

        except WebSocketDisconnect:
            print("Client disconnected")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"WebSocket error: {e}")
            try:
                await websocket.send_json({
                    "status": "error",
                    "message": str(e),
                })
            except:
                pass

    return app
