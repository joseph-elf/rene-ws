import asyncio
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from package import *

TEMPERATURE=1
K=5

BROADCAST_FREQUENCY = 2
WORD_SIZE = 5

DELAY = (1 / BROADCAST_FREQUENCY)
TOKENS_PER_SECOND = WORD_SIZE*BROADCAST_FREQUENCY

MAX_BUFFER = 500



# Build the LLM
device = torch.device("cpu")
model = GPT.load("models/gpt-sc-6-cpu.w").to(device)
tokenizer = SingleCharTokenizer()
_=tokenizer.load_tokens("models/tokens_sc.tok")

inferer = Inferer(model,tokenizer,device,buffer_size=MAX_BUFFER)
print("The model is loaded")


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



clients: set[WebSocket] = set()



broadcast_task = None
# ---------------------------------------------------
# BROADCAST LOOP
# ---------------------------------------------------

async def broadcaster():

    global broadcast_task

    try:
        while True:

            # stop generation if nobody watching
            if len(clients) == 0:
                break

            start = time.time()

            word = inferer.generate_next_word(WORD_SIZE,TEMPERATURE,K)

            dead = []

            for ws in clients:
                try:
                    await ws.send_text(word)
                except:
                    dead.append(ws)

            for ws in dead:
                clients.remove(ws)

            elapsed = time.time() - start
            await asyncio.sleep(max(0, DELAY - elapsed))

    finally:
        broadcast_task = None


# ---------------------------------------------------
# WEBSOCKET
# ---------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):

    global broadcast_task

    await ws.accept()
    clients.add(ws)

    # send buffer
    #for token in inferer.buffer:
    await ws.send_text(inferer.buffer)

    # start generation if first client
    if broadcast_task is None:
        broadcast_task = asyncio.create_task(broadcaster())

    try:
        while True:
            await asyncio.sleep(60)

    except WebSocketDisconnect:

        clients.remove(ws)
