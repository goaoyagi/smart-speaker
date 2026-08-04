#!/usr/bin/env python3
"""
ハードウェアテストモジュール - LEDとボタンの物理配線の確認用。

Raspberry Pi 上で手動実行し、ボタンを押している間LEDが点灯すれば配線は正常。
ピン番号は src/config.py（および .env）から読むため、アプリ本体が実際に使う
ピンをそのまま検証できる。
"""

import sys
from pathlib import Path
from signal import pause

# プロジェクトルートを sys.path に追加し、src/config.py を読めるようにする。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gpiozero import Button, LED  # noqa: E402

from src.config import PTT_BUTTON_PIN, STATUS_LED_PIN  # noqa: E402

# ボタンはGPIOとGNDの間に接続する（内蔵プルアップを使用し、押した時にLOWになる）
button = Button(PTT_BUTTON_PIN, pull_up=True)
led = LED(STATUS_LED_PIN)

print("=========================================")
print("  スマートスピーカー ハードウェアテストモジュール")
print("=========================================")
print(f"▶ ボタン: GPIO {PTT_BUTTON_PIN} / LED: GPIO {STATUS_LED_PIN}")
print("▶ 状態: 待機中...")
print("▶ アクション: ブレッドボードのボタンを押してみてください。")
print("※ 終了するには [Ctrl + C] を押してください。\n")

# イベントと処理の紐付け（イベント駆動）
button.when_pressed = led.on
button.when_released = led.off

# プログラムが勝手に終了しないようにストップさせておく
try:
    pause()
except KeyboardInterrupt:
    print("\nテストを安全に終了しました。お疲れ様でした！")
