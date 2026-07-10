# AIスマートスピーカー 要件・コンテキスト（ラズパイ5）

> セットアップ・実行方法は `README.md`、開発ルールは `AGENTS.md` を参照。

## 1. プロジェクト概要

ラズパイ 5 上で動作する**ローカル完結型**の日本語 AI スマートスピーカー。  
外部クラウド API を使わず、すべてのコンポーネント（音声認識・検索・LLM・TTS）をローカルで完結させる。

## 2. 設計思想：生成前 RAG によるハルシネーション防止

LLM が「知らないことを作り話する（ハルシネーション）」を根本から防ぐため、**「生成後の検証」ではなく「生成前 RAG」**を採用する。  
ユーザーの質問を受け取ったら、まず Web 検索で事実を収集し、その検索結果をプロンプトにそのまま埋め込んでから LLM に渡す。LLM は与えられた事実のみをもとに回答する。

## 3. システムアーキテクチャ（5ステップパイプライン）

```
マイク入力
   │
   ▼
[耳] listener.py   — arecord で録音 → faster-whisper で日本語テキスト化
   │
   ▼
[検索] retriever.py  — SearXNG（localhost:8080）でWeb検索、上位5件取得
   │
   ▼
[構成] composer.py   — 検索結果＋質問をプロンプトに編成（日本語のみ出力を強制）
   │
   ▼
[脳] brain.py      — Ollama（localhost:11434、Qwen2.5:3b）でレスポンス生成
   │
   ▼
[口] speaker.py    — piper-tts-plus で音声合成 → aplay で再生
```

`status_led.py` は各ステップに連動して GPIO LED 状態（IDLE / LISTENING / SEARCHING / THINKING / SPEAKING / ERROR）を更新し、視覚フィードバックを提供する。

## 4. 技術選定の理由

| コンポーネント | 採用技術 | 理由 |
|---|---|---|
| 音声認識 | faster-whisper（small モデル） | ラズパイ 5 で CPU 推論可能な精度・速度バランス |
| Web 検索 | SearXNG（ローカル Docker） | プライバシー保護・外部 API 不要のローカル検索 |
| LLM | Ollama + Qwen2.5:3b | ラズパイ 5 のメモリ内で動作する最小クラスの高品質モデル |
| TTS | piper-tts-plus | 本家 piper（espeak-ng 依存）は日本語アクセント解析未対応のため、日本語特化フォーク版を採用 |
| TTS モデル | 日本語 ONNX モデル（JVS データセット等） | Hugging Face 等から取得して `models/` に配置 |

## 5. 将来拡張バックログ

詳細は `future_extensions.md` を参照。

- **ウェイクワード起動**（OpenWakeWord / Porcupine）— ボタン不要の常時待機
- **ローカルドキュメント検索**（ChromaDB / FAISS）— プライベート RAG の追加
- **会話コンテキスト保持**（`conversation_history.py` + sliding window memory）— マルチターン対話
