# WhisperLiveKit 測試數據總表(2026-08 優化系列)

彙整 2026-08-13 → 08-26 期間所有量測:測試資料、WER、翻譯品質(chrF++/COMET)、
TranslateGemma 吞吐(tok/s)、decode step 延遲、以及各拓樸的最大穩定連線數。
逐項細節見同目錄的 `2026-08-16_59bba9a_optimization.md`、
`2026-08-20_batchdecode_optimization.md`、`2026-08-13_71f7ff7_translate-debug_quant-sweep.md`。

## 0. 測試環境

- 主機:2× NVIDIA H200 NVL(143.8 GB VRAM、SM 9.0)、192 核、1 TB RAM。
  乾淨壓測一律在 GPU 1(GPU 0 常被其他使用者佔用)。
- ASR 模型:`models/cool-whisper`(distil 架構:CT2 fp16 encoder 32 層 +
  PyTorch fp32 decoder **2 層**,`--language zh`)。
- 翻譯:TranslateGemma(`Infomaniak-AI/vllm-translategemma-*-it`,vLLM 0.27.1,
  `MAX_MODEL_LEN=512`、ngram speculative decoding)。
- 量測期間 build:`59bba9a`(08-16 CPU/GIL 優化)→ `2fcfba4`(08-20/21
  跨 session batched decode + CUDA graph,現行 HEAD)。

## 1. 測試資料

| 用途 | 資料 | 內容 |
|---|---|---|
| WER(`--youtube-debug`) | `dataset/PLCX-BLZ1hDpDOgZPSmdMcpgfO5uQ0i4XK/test1/` | NTU OCW 課程影片 1 部(`-zD50IAAxfY`,600 s)+ 人工 `zh-TW.srt` 參考 |
| 翻譯(`--translate-debug`) | `dataset/PLCX-BLZ1hDpDOgZPSmdMcpgfO5uQ0i4XK/` | 同播放清單 4 部(600–1591 s),zh→en;對齊後 73 個 segment window,參考 `en.srt` |
| 壓測 fixture | 上述 4 部 wav → 16 kHz s16le PCM | 每連線輪流取一檔、起點偏移 61×idx 秒、0.5 s chunk 即時推送;檔案與腳本存 `benchmarking/results/loadtest_scripts/` |
| TG 吞吐 probe | 講課 zh 句(校準集之外 held-out) | zh→en,~48 output tokens/request |

壓測判準:600 s、**0 次 lag catch-up + 0 連線錯誤**才算過,頂點複測 ×2;
跑前必先小流量暖機。注意:fixture 必須用 4 檔輪流 —— 單檔會讓特定連線
起點落在 ~9.5 s 非語音段,在 10 s 門檻邊緣造成假觸發(詳見 08-20 報告
methodology note)。

## 2. WER(youtube-debug、max-speed、`--accumulate-mode lines`)

| Build | WER |
|---|---|
| 優化前 baseline(da7cf1a,×3) | 0.0608–0.0612 |
| 08-16 優化後(59bba9a,×3) | 0.0608 / 0.0592 / 0.0621(噪音帶 0.0592–0.0621) |
| batched decode 中間版 | 0.0608 / 0.0604 |
| **現行 build(2fcfba4,×2)** | **0.0612 / 0.0612** — 帶內,無回歸 |

Batched decode 的數值等價性另以單元測試驗證:batched vs 原始路徑
max |Δlogits| = 2.8e-9,argmax 完全相同。

## 3. 翻譯品質(translate-debug;chrF++ 受 ASR 切分噪音影響大,以 COMET 為準)

### 3.1 量化掃描(08-13,12b 為當時生產模型)

| 模型 | 量化 | 最小 VRAM | chrF++ | COMET |
|---|---|---|---|---|
| 4b | bf16 / fp8 / NF4 | 12.4 / 9.8 / 7.7 GiB | 52.2 / 53.1 / 50.3 | 0.7173 / 0.7172 / 0.7053 |
| 12b | bf16 / fp8 / NF4 | 27.6 / 17.5 / 12.6 GiB | 51.7 / 56.1 / 52.3 | 0.7258 / 0.7270 / 0.7215 |
| 27b | bf16 / **fp8** / NF4 | 56.2 / **32.4** / 21.7 GiB | 54.5 / **58.0** / 47.7 | 0.7307 / **0.7305** / 0.7178 |

→ fp8(Q8)在所有尺寸都是「免費」壓縮;27b Q8 為品質天花板,**已選用**。

### 3.2 NVFP4 27b 評估(08-26,同 build 同日配對)

| 27b 變體 | chrF++ | COMET |
|---|---|---|
| Q8 fp8(配對跑) | 56.96 | 0.7372 |
| NVFP4 run1 / run2 | 50.69 / 51.86 | 0.7358 / 0.7343 |

品質等同(Δ 在 ±0.0066 帶內),但 H200 無原生 FP4,vLLM 退化為 Marlin
W4A16:速度無差、GPU 佔用反增(vLLM SM 36→42%)、上限不變 → **不採用**。

## 4. TranslateGemma 吞吐 / 延遲(27b,H200,直接打 vLLM)

| 併發 | fp8:p50 / p90 / tok·s⁻¹ | NVFP4:p50 / p90 / tok·s⁻¹ |
|---|---|---|
| 1 | 756 / 1064 ms / **63** | 739 / 961 ms / 65 |
| 8 | 761 / 1082 ms / **284** | 767 / 1148 ms / 282 |
| 16 | 859 / 1221 ms / **465** | 878 / 1171 ms / 494 |

壓測全程 vLLM `waiting=0` —— 翻譯端從未排隊,不是瓶頸。

## 5. Decode step 延遲(batched executor,cool-whisper 2 層 decoder)

| 量測 | 數值 |
|---|---|
| 原始 batch-1 step(單獨) | ~1.14 ms/step(幾乎全是 CPU dispatch) |
| batched + CUDA graph step busy(單獨) | **1.0–1.1 ms**(第一版 eager 6.5 ms) |
| N=1 經 executor 全程往返 | 1.36 ms/step(+0.22 ms,兩次 thread hop;體感無差) |
| 高載下 step busy | 6–9 ms(GIL 排隊 / GPU timeline 排隊所致,非 kernel 變慢) |

## 6. 最大穩定連線數(600 s、0 觸發 + 0 錯誤;單一 H200 同卡跑 WLK+TG)

### 演進(2×WLK + 1×TG)

| Build | TG | 上限 |
|---|---|---|
| 優化前(08 月初) | 12b fp8 | 13 |
| 08-16 CPU/GIL 優化(59bba9a) | 12b fp8 | 17 |
| 08-20 batched decode(2fcfba4) | 12b fp8 | **19**(20 失敗 6+0) |
| 同上 | 27b fp8 | **16**(17 失敗 3+0) |
| 同上 | 27b NVFP4 | 16(17 失敗 12+0)— 無增益 |

### 27b fp8(選用配置)各拓樸(08-21,4 檔 fixture)

| 拓樸 | 上限 | 失敗樣態 | 失敗時 GPU |
|---|---|---|---|
| 1×WLK + TG | **13**(14 邊緣 1/2) | 15 → 13 次觸發,單 process 卡死 | 76%(Python/GIL bound) |
| 2×WLK + TG | **16** | 17 → 3+0 | ~89%(GIL+GPU 疊加) |
| 3×WLK + TG | **18**(19 邊緣 2/3) | 20 → 0+14+0 | 94%(GPU timeline bound) |

VRAM 峰值(3×WLK+27b、N=20):wlk 各 ~9 GiB + vLLM 47.8 GiB ≈ 75.7 GiB
(佔整卡 54%)—— VRAM 從不是限制。

## 7. 結論速覽

- 現行部署選擇:**27b Q8(online fp8)**;batched decode 預設開
  (`--no-batch-decode` 可關、`--batch-decode-slots 16`)。
- 單 process 上限 ~13–14 條(GIL);多 process 攤掉 GIL 後,單卡上限
  ~18–19 條(GPU timeline,vLLM burst + 3×WLK 小 kernel 塞滿時間軸)。
- 再往上的選項:TG 移到第二張卡、或第二張卡再開一組
  (2 卡估 ~36–38 條);NVFP4 需 Blackwell 才有意義。
