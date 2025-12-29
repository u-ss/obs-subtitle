"""
OBS AI Studio LOCAL ULTIMATE v2
-----------------------------------------------------
【修正点】
1. 「話の区切り(秒)」スライダーを追加。
   → 途中で切れる場合は、これを「1.5」や「2.0」に増やしてください。
2. マイク感度と区切り判定をリアルタイムで調整可能に。
3. これまでの機能（ローカルWhisper、幻覚除去、辞書補正）は全て搭載。

必須: pip install faster-whisper openai python-dotenv speechrecognition pyaudio
"""

import os
import sys
import threading
import queue
import time
import datetime
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# ライブラリ読み込み
try:
    from dotenv import load_dotenv
    import speech_recognition as sr
    from openai import OpenAI
    from faster_whisper import WhisperModel
except ImportError as e:
    print(f"起動エラー: ライブラリ不足\n{e}")
    sys.exit(1)

# -----------------------------------------------------------------------------
# 1. 設定
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.resolve()
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("OPENAI_API_KEY")

# 辞書
CUSTOM_VOCAB = "RADWIMPS, World's End Girlfriend, indigo la End, シリウスループ, 夏夜のマジック, 明け星, VOICEBOX, DAD, OBS, Python, GPT-4o, VTuber"

# 幻覚ブロック (会話の「字幕」は通す設定)
HALLUCINATION_PHRASES = [
    "ご視聴ありがとうございました", "チャンネル登録", "おいしいですね", 
    "Thank you for watching", "Subtitles by", "Amara.org",
    "和訳をお願いします", "翻訳をお願いします", "視聴ありがとうございました",
    "字幕作成", "字幕 :"
]

# Whisper設定
MODEL_SIZE = os.getenv("WHISPER_MODEL", "large-v3")
DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE", "float16")

# ファイルパス
FILE_RAW = Path(os.getenv("SUB_JP_RAW", r"C:\obs\caption_jp_raw.txt"))
FILE_MILD = Path(os.getenv("SUB_JP", r"C:\obs\caption_jp.txt"))
FILE_TRANS = Path(os.getenv("SUB_JP_TRANS", r"C:\obs\caption_jp_trans.txt"))

# GPTプロンプト
PROMPT_MILD = """
あなたは「字幕の修正係」です。
入力されたテキストは、音声認識AIが聞き取ったリストです。以下のルールで修正してください。

【重要ルール】
1. **固有名詞の推測修正:**
   文脈から判断して、明らかな聞き間違いを正しい表記に直してください。
   例: "ラットヴィンペス" → "RADWIMPS"
   例: "ワールドエンドガールフレンド" → "World's End Girlfriend"
   例: "インディゴラ、エンド" → "indigo la End"

2. **口調の維持:**
   話者の口調、方言、語尾（～やな、～よな）、フィラーは修正せず、そのまま残してください。

3. **幻覚の削除:**
   「和訳をお願いします」「チャンネル登録」などの文脈に関係ないフレーズは削除してください。
"""

PROMPT_TRANS = """
あなたは同時通訳のサポートAIです。
入力されたテキストを、翻訳ソフトが誤訳しないよう「主語を補った、簡潔な標準語（です・ます調）」にリライトしてください。
"""

# -----------------------------------------------------------------------------
# 2. AI エンジン
# -----------------------------------------------------------------------------
class LocalAIEngine:
    def __init__(self, api_key, gui_callback):
        self.client = OpenAI(api_key=api_key)
        self.callback = gui_callback
        self.running = False
        
        # マイク設定
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        
        # 自動調整OFF（手動スライダーで制御するため）
        self.recognizer.dynamic_energy_threshold = False
        
        # 初期値
        self.recognizer.energy_threshold = 300   # 感度
        self.recognizer.pause_threshold = 1.2    # ★ここが「区切り」の秒数

        self.audio_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.whisper = None

    def set_energy_threshold(self, value):
        """マイク感度の変更"""
        try:
            self.recognizer.energy_threshold = float(value)
        except:
            pass

    def set_pause_threshold(self, value):
        """区切り(無音)時間の変更"""
        try:
            val = float(value)
            self.recognizer.pause_threshold = val
            # non_speaking_durationも少し連動させると安定する
            self.recognizer.non_speaking_duration = val * 0.8
        except:
            pass

    def load_model(self):
        self.callback("sys", f"モデル読込中({MODEL_SIZE} / {DEVICE})...")
        try:
            self.whisper = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
            self.callback("sys", "モデルロード完了！")
            return True
        except Exception as e:
            self.callback("sys", f"モデルロード失敗: {e}\nDLL不足の可能性があります。")
            return False

    def start(self):
        self.running = True
        if self.whisper is None:
            if not self.load_model():
                self.running = False
                return

        try:
            with self.mic as source:
                self.callback("sys", "マイク初期化中...")
            
            self.callback("sys", f"辞書設定: {CUSTOM_VOCAB[:20]}...")
            self.callback("status", "待機中")

            # バックグラウンドリッスン開始
            self.stop_listening = self.recognizer.listen_in_background(
                self.mic, self._on_audio_captured, phrase_time_limit=20
            )
            threading.Thread(target=self._process_queue, daemon=True).start()
        except Exception as e:
            self.callback("sys", f"起動エラー: {e}")

    def stop(self):
        self.running = False
        if hasattr(self, 'stop_listening'):
            self.stop_listening(wait_for_stop=False)
        self.executor.shutdown(wait=False)

    def _on_audio_captured(self, recognizer, audio):
        if self.running:
            self.callback("status", "音声受信 → Local推論中...")
            self.audio_queue.put(audio)

    def _process_queue(self):
        while self.running:
            try:
                audio = self.audio_queue.get(timeout=0.1)
                self._pipeline(audio)
            except queue.Empty:
                continue

    def _pipeline(self, audio_data):
        try:
            # Whisper
            import io
            wav_bytes = audio_data.get_wav_data()
            wav_stream = io.BytesIO(wav_bytes)
            
            segments, info = self.whisper.transcribe(
                wav_stream, 
                beam_size=5, 
                language="ja",
                initial_prompt=CUSTOM_VOCAB,
                vad_filter=True, 
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            raw_text = "".join([segment.text for segment in segments]).strip()

            if len(raw_text) < 2: return 

            # 幻覚フィルタ
            for h in HALLUCINATION_PHRASES:
                if h in raw_text:
                    self.callback("sys", f"幻覚ブロック: {raw_text}")
                    self.callback("status", "待機中")
                    return

            self.callback("raw", raw_text)
            self._write_file(FILE_RAW, raw_text)

            # GPT
            future_mild = self.executor.submit(self._gpt_req, PROMPT_MILD, raw_text)
            future_trans = self.executor.submit(self._gpt_req, PROMPT_TRANS, raw_text)

            mild_text = future_mild.result()
            trans_text = future_trans.result()

            if mild_text:
                self.callback("mild", mild_text)
                self._write_file(FILE_MILD, mild_text)
            
            if trans_text:
                self.callback("trans", trans_text)
                self._write_file(FILE_TRANS, trans_text)

            self.callback("status", "待機中")

        except Exception as e:
            self.callback("sys", f"Error: {e}")

    def _gpt_req(self, system_prompt, user_text):
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text}
                ],
                temperature=0.2 
            )
            return resp.choices[0].message.content.strip()
        except:
            return ""

    def _write_file(self, path: Path, text: str):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        except:
            pass

# -----------------------------------------------------------------------------
# 3. GUI
# -----------------------------------------------------------------------------
class MainGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("OBS AI Studio LOCAL v2 (区切り調整版)")
        self.geometry("1000x600")
        self.minsize(800, 450)
        self.configure(bg="#2b2b2b")
        self.engine = None
        self._init_ui()

    def _init_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background="#2b2b2b")
        style.configure("TLabel", background="#2b2b2b", foreground="white")
        style.configure("TButton", font=("Meiryo UI", 10, "bold"))

        # ヘッダーエリア
        header = ttk.Frame(self, padding=10)
        header.pack(fill=tk.X, side=tk.TOP)
        
        # 左側: ボタンとステータス
        left_box = ttk.Frame(header)
        left_box.pack(side=tk.LEFT)
        self.btn_start = ttk.Button(left_box, text="SYSTEM START", command=self._toggle)
        self.btn_start.pack(side=tk.LEFT, padx=(0, 15))
        self.lbl_status = ttk.Label(left_box, text="Ready", foreground="#00ff00")
        self.lbl_status.pack(side=tk.LEFT)

        # 右側: スライダー2つ
        right_box = ttk.Frame(header)
        right_box.pack(side=tk.RIGHT)

        # 1. 感度スライダー
        f_sens = ttk.Frame(right_box)
        f_sens.pack(side=tk.LEFT, padx=15)
        ttk.Label(f_sens, text="マイク感度").pack(anchor="w")
        self.scale_sens = tk.Scale(f_sens, from_=300, to=3000, orient=tk.HORIZONTAL, 
                                   bg="#2b2b2b", fg="white", highlightthickness=0, showvalue=0, length=150,
                                   command=self._on_sens_change)
        self.scale_sens.set(300)
        self.scale_sens.pack()
        self.lbl_sens_val = ttk.Label(f_sens, text="300 (高感度)", font=("Meiryo UI", 8))
        self.lbl_sens_val.pack()

        # 2. 区切りスライダー (Pause Threshold)
        f_pause = ttk.Frame(right_box)
        f_pause.pack(side=tk.LEFT, padx=15)
        ttk.Label(f_pause, text="区切り(秒)").pack(anchor="w")
        self.scale_pause = tk.Scale(f_pause, from_=0.5, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, 
                                   bg="#2b2b2b", fg="white", highlightthickness=0, showvalue=0, length=150,
                                   command=self._on_pause_change)
        self.scale_pause.set(1.2) # 初期値
        self.scale_pause.pack()
        self.lbl_pause_val = ttk.Label(f_pause, text="1.2秒 (標準)", font=("Meiryo UI", 8))
        self.lbl_pause_val.pack()

        # メインエリア
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        paned = tk.PanedWindow(main_frame, orient=tk.HORIZONTAL, bg="#2b2b2b", sashwidth=4, sashrelief=tk.RAISED)
        paned.pack(fill=tk.BOTH, expand=True)

        self.txt_raw = self._create_col(paned, "1. Local Whisper (原文)", "#ffd700")
        self.txt_mild = self._create_col(paned, "2. Mild (AI補正)", "#00ffff")
        self.txt_trans = self._create_col(paned, "3. Trans (翻訳用)", "#ff00ff")

    def _create_col(self, parent, title, color):
        f = tk.Frame(parent, bg="#333333")
        parent.add(f, minsize=100, stretch="always")
        tk.Label(f, text=title, fg=color, bg="#333333", font=("Meiryo UI",10,"bold")).pack(anchor="w", padx=2, pady=2)
        t = scrolledtext.ScrolledText(f, bg="#1e1e1e", fg="white", font=("Meiryo UI",10), bd=0)
        t.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        return t

    def _on_sens_change(self, val):
        v = int(float(val))
        self.lbl_sens_val.configure(text=f"{v}")
        if self.engine: self.engine.set_energy_threshold(v)

    def _on_pause_change(self, val):
        v = float(val)
        self.lbl_pause_val.configure(text=f"{v}秒")
        if self.engine: self.engine.set_pause_threshold(v)

    def _toggle(self):
        if self.engine and self.engine.running:
            self.engine.stop()
            self.btn_start.configure(text="SYSTEM START")
            self.lbl_status.configure(text="Stopped", foreground="red")
        else:
            if not API_KEY:
                messagebox.showerror("Error", ".env Error")
                return
            self.engine = LocalAIEngine(API_KEY, self._update)
            # 初期値を適用
            self.engine.set_energy_threshold(self.scale_sens.get())
            self.engine.set_pause_threshold(self.scale_pause.get())
            
            self.engine.start()
            self.btn_start.configure(text="STOP")
            self.lbl_status.configure(text="Loading Model...", foreground="#ffff00")

    def _update(self, type_, text):
        self.after(0, lambda: self._do_update(type_, text))

    def _do_update(self, type_, text):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        if type_ == "sys":
            self.lbl_status.configure(text=text)
            self.txt_raw.insert(tk.END, f"[{ts}] [SYS] {text}\n")
            self.txt_raw.see(tk.END)
        elif type_ == "status":
            self.lbl_status.configure(text=text)
        elif type_ in ["raw","mild","trans"]:
            t = getattr(self, f"txt_{type_}")
            t.insert(tk.END, f"[{ts}] {text}\n\n")
            t.see(tk.END)

if __name__ == "__main__":
    app = MainGUI()
    app.mainloop()