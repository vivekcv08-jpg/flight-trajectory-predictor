import asyncio
import json
import subprocess
import time
import base64
from pathlib import Path
import urllib.request
import websockets

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
SCREENSHOT_DIR = Path(r"C:\Users\ACER\.gemini\antigravity\brain\2bdcbee9-5c07-49f4-8f7e-0581fecf776f")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOT_FILE = SCREENSHOT_DIR / "flight_predictor_verified.png"

async def run_browser_verification():
    # 1. Start Chrome headless with remote debugging
    cmd = [
        CHROME_PATH,
        "--headless=new",
        "--remote-debugging-port=9222",
        "--disable-gpu",
        "--window-size=1400,900",
        "--no-first-run",
        "--no-default-browser-check",
        "http://127.0.0.1:8000"
    ]
    proc = subprocess.Popen(cmd)
    try:
        # Wait for debugging port
        ws_url = None
        for _ in range(20):
            try:
                with urllib.request.urlopen("http://127.0.0.1:9222/json", timeout=2) as resp:
                    data = json.loads(resp.read().decode())
                    for page in data:
                        if page.get("type") == "page" and "8000" in page.get("url", ""):
                            ws_url = page.get("webSocketDebuggerUrl")
                            break
                    if not ws_url and len(data) > 0:
                        ws_url = data[0].get("webSocketDebuggerUrl")
                    if ws_url:
                        break
            except Exception:
                pass
            time.sleep(0.5)

        if not ws_url:
            raise RuntimeError("Could not connect to Chrome DevTools port 9222")

        print(f"Connected to CDP WebSocket: {ws_url}")

        async with websockets.connect(ws_url) as ws:
            msg_id = 0
            async def send_cmd(method, params=None):
                nonlocal msg_id
                msg_id += 1
                payload = {"id": msg_id, "method": method, "params": params or {}}
                await ws.send(json.dumps(payload))
                while True:
                    res = json.loads(await ws.recv())
                    if res.get("id") == msg_id:
                        return res

            await send_cmd("Page.enable")
            await send_cmd("Runtime.enable")

            # Wait 3 seconds for initial render and data fetch
            await asyncio.sleep(3)

            # Step 1: Check initial page state
            eval_res = await send_cmd("Runtime.evaluate", {
                "expression": "document.title"
            })
            print("Page Title:", eval_res.get("result", {}).get("result", {}).get("value"))

            # Step 2 & 3: Apply 'High Drag' preset and enable ODE overlay, then await predict()
            print("Applying 'High Drag' preset (250 m/s, 45 deg, 1.5 kg) and enabling ODE overlay...")
            eval_res = await send_cmd("Runtime.evaluate", {
                "expression": """
                (async () => {
                    applyPreset(250, 45, 1.5);
                    document.getElementById('ode-toggle').checked = true;
                    await predict();
                    return {
                        v0: document.getElementById('v0').value,
                        ang: document.getElementById('ang').value,
                        mass: document.getElementById('mass').value,
                        odeChecked: document.getElementById('ode-toggle').checked,
                        range: document.getElementById('kpi-range').textContent,
                        alt: document.getElementById('kpi-alt').textContent,
                        audit: document.getElementById('audit-content').innerText
                    };
                })()
                """,
                "awaitPromise": True,
                "returnByValue": True
            })
            print("Eval result:", json.dumps(eval_res, indent=2))

            # Read back values from UI to verify
            eval_res = await send_cmd("Runtime.evaluate", {
                "expression": """
                JSON.stringify({
                    badgeInf: document.getElementById('badge-inf').textContent,
                    badgeSpeed: document.getElementById('badge-speed').textContent,
                    kpiRange: document.getElementById('kpi-range').textContent,
                    kpiAlt: document.getElementById('kpi-alt').textContent,
                    auditVisible: document.getElementById('audit-box').classList.contains('visible'),
                    auditText: document.getElementById('audit-content').innerText
                })
                """
            })
            ui_state = json.loads(eval_res.get("result", {}).get("result", {}).get("value", "{}"))
            print("UI Verified State:", json.dumps(ui_state, indent=2))

            # Step 4: Capture Screenshot
            print("Capturing full screenshot...")
            ss_res = await send_cmd("Page.captureScreenshot", {"format": "png"})
            b64_data = ss_res.get("result", {}).get("data", "")
            if b64_data:
                SCREENSHOT_FILE.write_bytes(base64.b64decode(b64_data))
                print(f"Screenshot successfully saved to: {SCREENSHOT_FILE}")
            else:
                print("Failed to capture screenshot data.")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()

if __name__ == "__main__":
    asyncio.run(run_browser_verification())
