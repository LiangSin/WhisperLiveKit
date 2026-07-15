# Per-Connection Session Config（連線層級的客製化 Prompt）

每一條 `/asr` WebSocket 連線都可以在**送出音訊之前**，先送一個 JSON 文字訊息來客製化這個 session 的轉錄行為。目前支援的欄位是 Whisper 的 prompt（用來提升專有名詞、術語的辨識準確度）。

不送 config 訊息的舊客戶端完全不受影響（向下相容）：連上後直接送音訊 bytes 即可，行為與過去相同，prompt 沿用伺服器啟動時的 `--init-prompt` / `--static-init-prompt` 預設值。

> 僅支援 SimulStreaming backend policy（預設）。若伺服器以 LocalAgreement policy 執行，config 訊息會被接受但 prompt 欄位會被忽略（伺服器 log 會有警告）。

---

## 訊息流程

```
Client                                Server
  │  ── WebSocket 連線 ──────────────→  │
  │  ←─ {"type":"config", ...} ───────  │   伺服器 hello（useAudioWorklet 等）
  │                                     │
  │  ── {"type":"config",              │
  │      "static_init_prompt":"..."} ─→ │   (選用) 必須在第一個音訊 chunk 之前
  │  ←─ {"type":"config_ack", ...} ───  │
  │                                     │
  │  ── 音訊 bytes (binary frames) ───→ │
  │  ←─ 轉錄結果 (JSON) ──────────────  │
  │  ── 空 binary frame (b"") ────────→ │   結束訊號
  │  ←─ {"type":"ready_to_stop"} ─────  │
```

重點規則：

1. **時機**：config 必須在第一個音訊 binary frame 之前送出。音訊開始後才送 config 會收到 `config_error`，且不會被套用。
2. **確認**：伺服器成功套用後回覆 `config_ack`，`applied` 欄位列出實際生效的值。建議等收到 ack 再開始送音訊。
3. **一次性**：config 對整條連線生效，中途不能更改。要換 prompt 就開新連線。

---

## Config 訊息格式

```jsonc
{
  "type": "config",                          // 必填，固定為 "config"
  "static_init_prompt": "台積電, ChatGPT",    // 選填：術語表，整段音訊都生效
  "init_prompt": "以下是科技新聞節目。"        // 選填：開場語境，會隨 context 滾動被擠掉
}
```

| 欄位 | 型別 | 說明 |
|---|---|---|
| `static_init_prompt` | string | 固定釘在 decoder context 開頭、**永遠不會被滾走**。適合放整段音訊都相關的專有名詞、人名、術語表。**大多數情況用這個。** |
| `init_prompt` | string | 只在開頭提供語境，隨著轉錄進行會被新的上文擠掉。適合描述音訊的開場情境。 |

欄位語義：

- **省略欄位或給 `null`** → 沿用伺服器的預設值（CLI 的 `--static-init-prompt` / `--init-prompt`）。
- **給空字串 `""`** → 明確清除伺服器預設值（這條連線不用 prompt）。
- **給非字串值** → 收到 `config_error`，整個 config 不套用。

### 伺服器回覆

```jsonc
// 成功
{"type": "config_ack", "applied": {"static_init_prompt": "台積電, ChatGPT"}}

// 失敗（太晚送、型別錯誤）
{"type": "config_error", "message": "config must be sent before the first audio chunk"}
```

未知欄位會被忽略（向前相容），不會出現在 `applied` 裡。

---

## Prompt 撰寫建議

- 用**目標語言**撰寫（中文轉錄就用中文 prompt），prompt 的語言與風格會影響輸出（例如簡體 prompt 可能導向簡體輸出）。
- Whisper 的 prompt 上限約 224 tokens，而且 `static_init_prompt` 會佔掉滾動上文的空間，**術語表越精簡效果越好**。
- 逗號分隔的詞彙表即可；若某個詞常被拼錯，用正確拼法造一個短句效果通常更強。
- 這是軟性偏置不是保證：詞太多會互相稀釋。

---

## Client 範例

### Python（websockets）

```python
import asyncio, json
import websockets

async def transcribe(audio_chunks, terminology):
    async with websockets.connect("ws://localhost:8000/asr") as ws:
        await ws.recv()  # 伺服器 hello: {"type":"config","useAudioWorklet":...}

        # 1. 送 session config，等 ack
        await ws.send(json.dumps({
            "type": "config",
            "static_init_prompt": terminology,
        }))
        ack = json.loads(await ws.recv())
        assert ack["type"] == "config_ack", ack

        # 2. 開始串流音訊（webm/opus 或伺服器設定的格式）
        async def send_audio():
            for chunk in audio_chunks:
                await ws.send(chunk)
                await asyncio.sleep(0.05)
            await ws.send(b"")  # 結束訊號

        sender = asyncio.create_task(send_audio())
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") == "ready_to_stop":
                break
            for line in msg.get("lines", []):
                print(line["text"])
        await sender
```

### JavaScript（瀏覽器）

```javascript
const ws = new WebSocket("ws://localhost:8000/asr");
let configAcked = false;

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: "config",
    static_init_prompt: "台積電, WhisperLiveKit, SimulStreaming",
  }));
};

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.type === "config_ack") {
    configAcked = true;
    startRecording();          // 收到 ack 後才開始送音訊
  } else if (msg.type === "config_error") {
    console.error("config rejected:", msg.message);
  } else if (msg.lines) {
    render(msg.lines);         // 轉錄結果
  }
};

// MediaRecorder ondataavailable → ws.send(blob)，與原本用法相同
```

---

## 實作位置（維護參考）

- 伺服器收訊與 config 解析：`whisperlivekit/basic_server.py` 的 `websocket_endpoint`（`AudioProcessor` 延後到第一個音訊 chunk 才建立，讓 config 有機會先到）。
- 參數傳遞：`AudioProcessor.__init__` → `online_factory()`（`whisperlivekit/core.py`）→ `SimulStreamingOnlineProcessor`。
- 生效機制：`SimulStreamingOnlineProcessor` 用 `dataclasses.replace()` 複製一份 session 專屬的 `AlignAttConfig`，所以不會污染其他連線；`AlignAtt.init_context()`（`simul_whisper.py`）在 session 自己的 `DecoderState` 裡套用 prompt，`trim_context()` 會保護 `static_init_prompt` 不被滾走。
