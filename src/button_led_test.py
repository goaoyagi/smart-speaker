from gpiozero import Button, LED
from signal import pause

# 1. ハードウェアのピン設定
# GPIO 24番にスイッチを接続（プルアップを有効にして、押した時にLOWになるよう設定）
button = Button(24, pull_up=True)

# GPIO 23番にLEDを接続
led = LED(23)

print("=========================================")
print("  スマートスピーカー ハードウェアテストモジュール")
print("=========================================")
print("▶ 状態: 待機中...")
print("▶ アクション: ブレッドボードのボタンを押してみてください。")
print("※ 終了するには [Ctrl + C] を押してください。\n")

# 2. イベントと処理の紐付け（イベント駆動）
# 「ボタンが押されたとき(when_pressed)」に「LEDを点灯(led.on)」させる
button.when_pressed = led.on

# 「ボタンが離されたとき(when_released)」に「LEDを消灯(led.off)」させる
button.when_released = led.off

# 3. プログラムが勝手に終了しないようにストップさせておく
try:
    pause()
except KeyboardInterrupt:
    print("\nテストを安全に終了しました。お疲れ様でした！")
