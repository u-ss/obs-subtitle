"""
OBS Subtitle Studio
Hallucination suppression, faster response, and usability tweaks.
"""

from __future__ import annotations

import io
import json
import os
import queue
import re
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

try:
    import numpy as np  # type: ignore
    import sounddevice as sd  # type: ignore

    SD_AVAILABLE = True
    SD_IMPORT_ERROR = ""
except Exception as e:  # pragma: no cover
    SD_AVAILABLE = False
    SD_IMPORT_ERROR = str(e)

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None

try:
    from openai import OpenAI  # type: ignore
except Exception as e:  # pragma: no cover
    OpenAI = None  # type: ignore
    OPENAI_IMPORT_ERROR = str(e)
else:
    OPENAI_IMPORT_ERROR = ""


APP_TITLE = "OBS Subtitle Studio"
CONFIG_FILE = "subtitle_studio_config.json"

DEFAULT_MILD_PROMPT = (
    "以下は音声認識結果です。話者の表現や言い回しはできるだけ残しつつ、"
    "誤変換や句読点を軽く整えてください。入力に存在しない内容・挨拶・締めの言葉"
    "（例: ご視聴ありがとうございました、どうぞよろしく 等）を追加しないでください。"
    "大きく書き換えないでください。\n"
    "----\n{transcript}\n----"
)

DEFAULT_TRANSLATE_PROMPT = (
    "以下はライブ配信の話者の発話です。翻訳前に、意味を保ったまま簡潔に整理し、"
    "誤変換を直し、短い文でまとめ直してください。入力に存在しない内容や締めの挨拶は"
    "絶対に追加しないでください。敬体/常体は元の雰囲気に合わせて構いません。\n"
    "箇条書きは使わず、自然な文章で1〜3文程度にまとめてください。\n"
    "----\n{transcript}\n----"
)


def _now() -> str:
    return time.strftime("%H:%M:%S")


def _bytesio_with_name(data: bytes, name: str) -> io.BytesIO:
    buf = io.BytesIO(data)
    buf.name = name  # type: ignore[attr-defined]
    return buf


def _rms_and_ratio(audio, threshold: int = 500) -> Tuple[float, float]:
    if getattr(np, "ndarray", None) is None or audio is None or len(audio) == 0:
        return 0.0, 0.0
    rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float32)))))
    voice_ratio = float(np.mean(np.abs(audio) > threshold))
    return rms, voice_ratio


def _norm_text(s: str) -> str:
    s2 = re.sub(r"[\s\u3000]+", "", s)
    s2 = re.sub(r"[、。．，,.!！?？・･]+", "", s2)
    return s2.lower()


def _default_patterns() -> list[re.Pattern[str]]:
    base = os.getenv(
        "HALLUCINATION_PATTERNS",
        "ご視聴ありがとうございました|チャンネル登録|高評価|ご覧いただき|登録よろしく|通知をオン|概要欄|サポートお願いします",
    )
    return [re.compile(p) for p in base.split("|") if p.strip()]


@dataclass
class TextBundle:
    basic: str
    mild: str
    summary: str
    mild_usage: Dict[str, int]
    summary_usage: Dict[str, int]
    timings: Dict[str, float]
    skipped: bool = False
    reason: str = ""


class SpeechProcessor:
    """Recording, VAD, Whisper, and GPT refinement."""

    def __init__(
        self,
        api_key: str,
        whisper_model: str = "whisper-1",
        gpt_model: str = "gpt-4o-mini",
        language: Optional[str] = None,
        min_rms: float = 15.0,
        min_voice_ratio: float = 0.12,
        vad_threshold: int = 500,
        min_text_chars: int = 6,
        patterns: Optional[list[re.Pattern[str]]] = None,
    ) -> None:
        if not OpenAI:
            raise RuntimeError(f"openai ライブラリを読み込めませんでした: {OPENAI_IMPORT_ERROR}")
        self.client = OpenAI(api_key=api_key)
        self.whisper_model = whisper_model
        self.gpt_model = gpt_model
        self.language = language or None
        self.min_rms = min_rms
        self.min_voice_ratio = min_voice_ratio
        self.vad_threshold = vad_threshold
        self.min_text_chars = min_text_chars
        self.patterns = patterns or _default_patterns()

    def record_chunk(self, seconds: float, samplerate: int = 16000) -> Tuple[bytes, Dict[str, float], Tuple[float, float]]:
        if not SD_AVAILABLE:
            raise RuntimeError(f"sounddevice を利用できません: {SD_IMPORT_ERROR}")
        t0 = time.perf_counter()
        frames = int(seconds * samplerate)
        audio = sd.rec(frames, samplerate=samplerate, channels=1, dtype="int16")
        sd.wait()
        t1 = time.perf_counter()
        rms, ratio = _rms_and_ratio(audio[:, 0], self.vad_threshold)
        with io.BytesIO() as buf:
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(samplerate)
                wf.writeframes(audio.tobytes())
            wav_bytes = buf.getvalue()
        timings = {"record_ms": (t1 - t0) * 1000}
        return wav_bytes, timings, (rms, ratio)

    def transcribe(self, wav_bytes: bytes) -> Tuple[str, Dict[str, float]]:
        t0 = time.perf_counter()
        resp = self.client.audio.transcriptions.create(
            model=self.whisper_model,
            file=_bytesio_with_name(wav_bytes, "audio.wav"),
            language=self.language,
        )
        text = getattr(resp, "text", None) or (resp.get("text") if isinstance(resp, dict) else "")
        return text or "", {"transcribe_ms": (time.perf_counter() - t0) * 1000}

    def _chat(self, prompt: str) -> Tuple[str, Dict[str, int], Dict[str, float]]:
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.gpt_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for live transcripts."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            timeout=20,
        )
        text = resp.choices[0].message.content or ""
        usage = {
            "prompt": getattr(resp.usage, "prompt_tokens", 0),
            "completion": getattr(resp.usage, "completion_tokens", 0),
            "total": getattr(resp.usage, "total_tokens", 0),
        }
        return text, usage, {"gpt_ms": (time.perf_counter() - t0) * 1000}

    def mild_cleanup(self, transcript: str, template: str = DEFAULT_MILD_PROMPT) -> Tuple[str, Dict[str, int], Dict[str, float]]:
        return self._chat(template.format(transcript=transcript.strip()))

    def translate_ready(self, transcript: str, template: str = DEFAULT_TRANSLATE_PROMPT) -> Tuple[str, Dict[str, int], Dict[str, float]]:
        return self._chat(template.format(transcript=transcript.strip()))

    def looks_hallucination(self, text: str, rms: float) -> bool:
        if not text:
            return False
        t = text.strip()
        if len(t) < self.min_text_chars:
            return True
        if rms < self.min_rms * 0.8 and len(t) < self.min_text_chars + 4:
            return True
        return any(p.search(t) for p in self.patterns)


class Worker(threading.Thread):
    """record -> VAD -> transcribe -> mild+summary (parallel) + filters."""

    def __init__(
        self,
        processor: SpeechProcessor,
        out_q: "queue.Queue[TextBundle]",
        stop_event: threading.Event,
        chunk_seconds: float,
        mild_template: str,
        summary_template: str,
        dup_suppress_sec: float,
    ) -> None:
        super().__init__(daemon=True)
        self.processor = processor
        self.out_q = out_q
        self.stop_event = stop_event
        self.chunk_seconds = chunk_seconds
        self.mild_template = mild_template
        self.summary_template = summary_template
        self.dup_suppress_sec = dup_suppress_sec
        self.last_norm = ""
        self.last_time = 0.0

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                wav, timing_rec, (rms, ratio) = self.processor.record_chunk(self.chunk_seconds)
                if rms < self.processor.min_rms or ratio < self.processor.min_voice_ratio:
                    self._emit_skip("silence", timing_rec, rms, ratio)
                    self._sleep_short()
                    continue

                basic, timing_tr = self.processor.transcribe(wav)
                if len(basic.strip()) < self.processor.min_text_chars:
                    self._emit_skip("short_text", {**timing_rec, **timing_tr}, rms, ratio)
                    self._sleep_short()
                    continue

                if self.processor.looks_hallucination(basic, rms=rms):
                    self._emit_skip("hallucination", {**timing_rec, **timing_tr}, rms, ratio)
                    self._sleep_short()
                    continue

                norm = _norm_text(basic)
                now = time.time()
                if norm and norm == self.last_norm and (now - self.last_time) < self.dup_suppress_sec:
                    self._emit_skip("duplicate", {**timing_rec, **timing_tr}, rms, ratio)
                    self._sleep_short()
                    continue

                t_chat_start = time.perf_counter()
                with ThreadPoolExecutor(max_workers=2) as exe:
                    fut_mild = exe.submit(self.processor.mild_cleanup, basic, self.mild_template)
                    fut_sum = exe.submit(self.processor.translate_ready, basic, self.summary_template)
                    mild_text, mild_usage, mild_t = fut_mild.result()
                    sum_text, sum_usage, sum_t = fut_sum.result()
                t_chat = (time.perf_counter() - t_chat_start) * 1000

                self.last_norm = norm
                self.last_time = now

                self.out_q.put(
                    TextBundle(
                        basic=basic,
                        mild=mild_text,
                        summary=sum_text,
                        mild_usage=mild_usage,
                        summary_usage=sum_usage,
                        timings={
                            **timing_rec,
                            **timing_tr,
                            "gpt_total_ms": t_chat,
                            "mild_ms": mild_t.get("gpt_ms", 0.0),
                            "summary_ms": sum_t.get("gpt_ms", 0.0),
                            "rms": rms,
                            "voice_ratio": ratio,
                        },
                    )
                )
            except Exception as e:  # pragma: no cover
                hint = ""
                if "model_not_found" in str(e) or "does not exist" in str(e):
                    hint = " (モデルが使えません。Whisperなら 'whisper-1' を試してください。)"
                self.out_q.put(
                    TextBundle(
                        basic=f"[{_now()}] エラー: {e}{hint}",
                        mild="",
                        summary="",
                        mild_usage={"prompt": 0, "completion": 0, "total": 0},
                        summary_usage={"prompt": 0, "completion": 0, "total": 0},
                        timings={},
                        skipped=True,
                        reason="error",
                    )
                )
            finally:
                self._sleep_short()

    def _emit_skip(self, reason: str, timings: Dict[str, float], rms: float, ratio: float) -> None:
        self.out_q.put(
            TextBundle(
                basic=f"[skip {reason}]",
                mild="",
                summary="",
                mild_usage={"prompt": 0, "completion": 0, "total": 0},
                summary_usage={"prompt": 0, "completion": 0, "total": 0},
                timings={**timings, "rms": rms, "voice_ratio": ratio},
                skipped=True,
                reason=reason,
            )
        )

    def _sleep_short(self) -> None:
        for _ in range(5):
            if self.stop_event.is_set():
                break
            time.sleep(0.12)


class SubtitleStudio(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1350x900")
        self.minsize(1100, 720)

        if load_dotenv:
            try:
                load_dotenv(override=False)
            except Exception:
                pass

        self.config_path = Path(CONFIG_FILE)
        self.cfg = self._load_cfg()

        self.api_key = os.getenv("OPENAI_API_KEY", self.cfg.get("api_key", ""))
        self.whisper_model = os.getenv("WHISPER_MODEL", self.cfg.get("whisper_model", "whisper-1"))
        self.gpt_model = os.getenv("GPT_MODEL", self.cfg.get("gpt_model", "gpt-4o-mini"))
        self.language = os.getenv("SUB_LANG", self.cfg.get("language", "")) or None
        self.chunk_seconds = float(os.getenv("CHUNK_SECONDS", self.cfg.get("chunk_seconds", 4.0)))
        self.mild_template = self.cfg.get("mild_template", DEFAULT_MILD_PROMPT)
        self.summary_template = self.cfg.get("summary_template", DEFAULT_TRANSLATE_PROMPT)

        self.min_rms = float(os.getenv("MIN_RMS", self.cfg.get("min_rms", 15.0)))
        self.min_voice_ratio = float(os.getenv("MIN_VOICE_RATIO", self.cfg.get("min_voice_ratio", 0.12)))
        self.vad_threshold = int(os.getenv("VAD_THRESHOLD", self.cfg.get("vad_threshold", 500)))
        self.min_text_chars = int(os.getenv("MIN_TEXT_CHARS", self.cfg.get("min_text_chars", 6)))
        self.dup_suppress_sec = float(os.getenv("DUP_SUPPRESS_SEC", self.cfg.get("dup_suppress_sec", 6.0)))

        self.wrap_var = tk.BooleanVar(value=bool(self.cfg.get("wrap", True)))
        self.font_size_var = tk.IntVar(value=int(self.cfg.get("font_size", 16)))
        self.topmost_var = tk.BooleanVar(value=bool(self.cfg.get("topmost", False)))

        self.mild_tokens = {"prompt": 0, "completion": 0, "total": 0}
        self.summary_tokens = {"prompt": 0, "completion": 0, "total": 0}

        self.stop_event = threading.Event()
        self.worker: Optional[Worker] = None
        self.out_q: "queue.Queue[TextBundle]" = queue.Queue()

        self._build_style()
        self._build_ui()
        self._apply_font()
        self._apply_topmost()
        self.after(120, self._poll_queue)

    def _load_cfg(self) -> dict[str, Any]:
        if self.config_path.exists():
            try:
                return json.loads(self.config_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_cfg(self) -> None:
        self.cfg["api_key"] = self.api_key
        self.cfg["whisper_model"] = self.whisper_model
        self.cfg["gpt_model"] = self.gpt_model
        self.cfg["language"] = self.language or ""
        self.cfg["chunk_seconds"] = self.chunk_seconds
        self.cfg["mild_template"] = self.mild_template
        self.cfg["summary_template"] = self.summary_template
        self.cfg["wrap"] = bool(self.wrap_var.get())
        self.cfg["font_size"] = int(self.font_size_var.get())
        self.cfg["topmost"] = bool(self.topmost_var.get())
        self.cfg["min_rms"] = self.min_rms
        self.cfg["min_voice_ratio"] = self.min_voice_ratio
        self.cfg["vad_threshold"] = self.vad_threshold
        self.cfg["min_text_chars"] = self.min_text_chars
        self.cfg["dup_suppress_sec"] = self.dup_suppress_sec
        try:
            self.config_path.write_text(json.dumps(self.cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _build_style(self) -> None:
        self.bg = "#0c1424"
        self.card = "#12263d"
        self.text_bg = "#0b1c30"
        self.text_fg = "#e8f0ff"
        self.accent = "#6dd5ff"

        self.configure(bg=self.bg)
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("TFrame", background=self.bg)
        style.configure("Card.TFrame", background=self.card)
        style.configure("TLabel", background=self.bg, foreground=self.text_fg, font=("Yu Gothic UI", 10))
        style.configure("Card.TLabel", background=self.card, foreground=self.text_fg, font=("Yu Gothic UI", 10))
        style.configure("Heading.TLabel", background=self.bg, foreground=self.accent, font=("Yu Gothic UI", 12, "bold"))
        style.configure("Section.TLabelframe", background=self.card, foreground=self.accent, padding=10)
        style.configure("Section.TLabelframe.Label", background=self.card, foreground=self.accent, font=("Yu Gothic UI", 11, "bold"))
        style.configure("Accent.TButton", background=self.accent, foreground="#0b1c30")
        style.map("Accent.TButton", background=[("active", "#8fe2ff")], foreground=[("active", "#0b1c30")])
        style.configure("TButton", background=self.card, foreground=self.text_fg)
        style.map("TButton", background=[("active", "#173554")], foreground=[("active", "#ffffff")])
        style.configure("Status.TLabel", background=self.bg, foreground="#9cb5d9")

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=(14, 12, 14, 12))
        root.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(root)
        header.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(header, text="OBS Subtitle Studio", style="Heading.TLabel").pack(side=tk.LEFT)
        ttk.Label(header, text="Whisper + GPT-4o-mini", style="TLabel").pack(side=tk.LEFT, padx=(10, 0))

        controls = ttk.Labelframe(root, text="操作と設定", style="Section.TLabelframe")
        controls.pack(fill=tk.X, pady=(0, 10))

        self.btn_start = ttk.Button(controls, text="開始 (録音 + 解析)", command=self._start, style="Accent.TButton")
        self.btn_start.grid(row=0, column=0, padx=6, pady=4, sticky="w")
        self.btn_stop = ttk.Button(controls, text="停止", command=self._stop, state=tk.DISABLED)
        self.btn_stop.grid(row=0, column=1, padx=6, pady=4, sticky="w")

        ttk.Label(controls, text="チャンク秒数").grid(row=0, column=2, padx=(20, 6), pady=4, sticky="e")
        self.chunk_var = tk.DoubleVar(value=self.chunk_seconds)
        ttk.Spinbox(controls, from_=3.0, to=20.0, increment=0.5, width=6, textvariable=self.chunk_var, command=self._on_chunk_change).grid(row=0, column=3, pady=4, sticky="w")

        ttk.Checkbutton(controls, text="折り返し", variable=self.wrap_var, command=self._on_wrap_toggle).grid(row=0, column=4, padx=12, pady=4, sticky="w")
        ttk.Checkbutton(controls, text="常に手前", variable=self.topmost_var, command=self._on_topmost_toggle).grid(row=0, column=5, padx=12, pady=4, sticky="w")

        ttk.Label(controls, text="フォント").grid(row=0, column=6, padx=(18, 4), pady=4, sticky="e")
        ttk.Spinbox(controls, from_=12, to=36, increment=1, width=5, textvariable=self.font_size_var, command=self._on_font_change).grid(row=0, column=7, pady=4, sticky="w")

        ttk.Button(controls, text="APIキー設定", command=self._set_api_key).grid(row=0, column=8, padx=(18, 6), pady=4, sticky="e")
        ttk.Button(controls, text="モデル設定", command=self._set_models).grid(row=0, column=9, padx=6, pady=4, sticky="e")
        ttk.Button(controls, text="テンプレ編集", command=self._edit_templates).grid(row=0, column=10, padx=6, pady=4, sticky="e")

        status_frame = ttk.Frame(root)
        status_frame.pack(fill=tk.X, pady=(0, 10))

        self.api_label = self._status_card(status_frame, "OpenAI API", self._api_status_text(), 0)
        self.model_label = self._status_card(status_frame, "モデル", f"Whisper: {self.whisper_model} / GPT: {self.gpt_model}", 1)
        self.token_label = self._status_card(status_frame, "トークン使用量", "mild: 0 | summary: 0", 2)

        main = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        self.txt_basic = self._make_pane(main, "1. ベーシック (Whisper)")
        self.txt_mild = self._make_pane(main, "2. マイルド整形 (GPT-4o-mini)")
        self.txt_summary = self._make_pane(main, "3. 翻訳用の要約/校正 (GPT-4o-mini)")

        token_box = ttk.Labelframe(root, text="APIトークン使用量", style="Section.TLabelframe")
        token_box.pack(fill=tk.X, pady=(0, 8))
        self.lbl_mild_tokens = ttk.Label(token_box, text="Mild prompt/completion/total: 0 / 0 / 0", style="Card.TLabel")
        self.lbl_mild_tokens.pack(anchor="w")
        self.lbl_summary_tokens = ttk.Label(token_box, text="Summary prompt/completion/total: 0 / 0 / 0", style="Card.TLabel")
        self.lbl_summary_tokens.pack(anchor="w")

        bottom = ttk.Labelframe(root, text="ログ / 遅延", style="Section.TLabelframe")
        bottom.pack(fill=tk.BOTH, expand=False)
        self.txt_log = ScrolledText(bottom, height=9, wrap=tk.WORD, bg=self.text_bg, fg=self.text_fg, insertbackground=self.text_fg)
        self.txt_log.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.txt_log.configure(state=tk.DISABLED)

        self.status = ttk.Label(self, text="準備完了", style="Status.TLabel")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def _status_card(self, parent: ttk.Frame, title: str, body: str, col: int) -> ttk.Label:
        frame = ttk.Frame(parent, style="Card.TFrame", padding=10)
        frame.grid(row=0, column=col, padx=(0 if col == 0 else 10, 0), sticky="nsew")
        ttk.Label(frame, text=title, style="Heading.TLabel").pack(anchor="w")
        lbl = ttk.Label(frame, text=body, style="Card.TLabel", wraplength=320)
        lbl.pack(anchor="w", pady=(4, 0))
        parent.grid_columnconfigure(col, weight=1)
        return lbl

    def _make_pane(self, parent: ttk.Panedwindow, title: str) -> ScrolledText:
        frame = ttk.Frame(parent, style="Card.TFrame", padding=8)
        parent.add(frame, weight=1)
        ttk.Label(frame, text=title, style="Card.TLabel", font=("Yu Gothic UI", 11, "bold"), foreground=self.accent).pack(anchor="w", pady=(0, 6))
        txt = ScrolledText(frame, wrap=tk.WORD if self.wrap_var.get() else tk.NONE, bg=self.text_bg, fg=self.text_fg, insertbackground=self.text_fg)
        txt.pack(fill=tk.BOTH, expand=True)
        txt.configure(state=tk.DISABLED)
        return txt

    def _start(self) -> None:
        if not self.api_key:
            messagebox.showwarning("APIキー未設定", "OPENAI_API_KEY を設定してください。")
            return
        if self.worker and self.worker.is_alive():
            return
        self.stop_event.clear()
        try:
            processor = SpeechProcessor(
                api_key=self.api_key,
                whisper_model=self.whisper_model,
                gpt_model=self.gpt_model,
                language=self.language,
                min_rms=self.min_rms,
                min_voice_ratio=self.min_voice_ratio,
                vad_threshold=self.vad_threshold,
                min_text_chars=self.min_text_chars,
            )
        except Exception as e:
            messagebox.showerror("初期化エラー", str(e))
            return
        self.worker = Worker(
            processor=processor,
            out_q=self.out_q,
            stop_event=self.stop_event,
            chunk_seconds=float(self.chunk_var.get()),
            mild_template=self.mild_template,
            summary_template=self.summary_template,
            dup_suppress_sec=self.dup_suppress_sec,
        )
        self.worker.start()
        self._log(f"[{_now()}] 録音と処理を開始しました")
        self.status.configure(text="処理中…")
        self.btn_start.configure(state=tk.DISABLED)
        self.btn_stop.configure(state=tk.NORMAL)

    def _stop(self) -> None:
        self.stop_event.set()
        self.btn_start.configure(state=tk.NORMAL)
        self.btn_stop.configure(state=tk.DISABLED)
        self.status.configure(text="停止しました")
        self._log(f"[{_now()}] 停止しました")

    def _on_chunk_change(self) -> None:
        try:
            self.chunk_seconds = float(self.chunk_var.get())
        except Exception:
            self.chunk_var.set(self.chunk_seconds)
        self._save_cfg()

    def _on_wrap_toggle(self) -> None:
        wrap = tk.WORD if self.wrap_var.get() else tk.NONE
        for w in (self.txt_basic, self.txt_mild, self.txt_summary, self.txt_log):
            w.configure(wrap=wrap)
        self._save_cfg()

    def _on_font_change(self) -> None:
        self._apply_font()
        self._save_cfg()

    def _apply_font(self) -> None:
        size = int(self.font_size_var.get())
        font = ("Yu Gothic UI", size) if os.name == "nt" else ("TkDefaultFont", size)
        for w in (self.txt_basic, self.txt_mild, self.txt_summary, self.txt_log):
            try:
                w.configure(font=font)
            except Exception:
                pass

    def _on_topmost_toggle(self) -> None:
        self._apply_topmost()
        self._save_cfg()

    def _apply_topmost(self) -> None:
        self.wm_attributes("-topmost", bool(self.topmost_var.get()))

    def _set_api_key(self) -> None:
        dlg = tk.Toplevel(self)
        dlg.title("APIキー設定")
        dlg.geometry("520x160")
        dlg.transient(self)
        dlg.grab_set()

        ttk.Label(dlg, text="OpenAI API Key").pack(anchor="w", padx=12, pady=(12, 4))
        var = tk.StringVar(value=self.api_key)
        ent = ttk.Entry(dlg, width=60, textvariable=var)
        ent.pack(anchor="w", padx=12, pady=4)
        ent.focus_set()

        def ok():
            self.api_key = var.get().strip()
            self._save_cfg()
            self._refresh_status_labels()
            dlg.destroy()

        ttk.Button(dlg, text="OK", command=ok).pack(anchor="e", padx=12, pady=10)

    def _set_models(self) -> None:
        dlg = tk.Toplevel(self)
        dlg.title("モデル設定")
        dlg.geometry("520x200")
        dlg.transient(self)
        dlg.grab_set()

        ttk.Label(dlg, text="Whisperモデル (音声→テキスト)").pack(anchor="w", padx=12, pady=(12, 4))
        w_var = tk.StringVar(value=self.whisper_model)
        ttk.Entry(dlg, width=50, textvariable=w_var).pack(anchor="w", padx=12, pady=4)

        ttk.Label(dlg, text="GPTモデル (マイルド/要約)").pack(anchor="w", padx=12, pady=(10, 4))
        g_var = tk.StringVar(value=self.gpt_model)
        ttk.Entry(dlg, width=50, textvariable=g_var).pack(anchor="w", padx=12, pady=4)

        ttk.Label(dlg, text="例) Whisper: whisper-1 / whisper-large-v3 が使えなければ whisper-1 を推奨").pack(anchor="w", padx=12, pady=(8, 4))

        def ok():
            self.whisper_model = w_var.get().strip() or "whisper-1"
            self.gpt_model = g_var.get().strip() or "gpt-4o-mini"
            self.model_label.configure(text=f"Whisper: {self.whisper_model} / GPT: {self.gpt_model}")
            self._save_cfg()
            dlg.destroy()

        ttk.Button(dlg, text="保存", command=ok).pack(anchor="e", padx=12, pady=10)

    def _edit_templates(self) -> None:
        dlg = tk.Toplevel(self)
        dlg.title("テンプレート編集")
        dlg.geometry("820x580")
        dlg.transient(self)
        dlg.grab_set()

        ttk.Label(dlg, text="マイルド整形プロンプト").pack(anchor="w", padx=10, pady=(10, 4))
        mild_box = ScrolledText(dlg, height=8, wrap=tk.WORD)
        mild_box.pack(fill=tk.X, padx=10, pady=(0, 10))
        mild_box.insert(tk.END, self.mild_template)

        ttk.Label(dlg, text="翻訳用 要約/校正プロンプト").pack(anchor="w", padx=10, pady=(0, 4))
        sum_box = ScrolledText(dlg, height=8, wrap=tk.WORD)
        sum_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        sum_box.insert(tk.END, self.summary_template)

        def apply():
            self.mild_template = mild_box.get("1.0", tk.END).strip() or DEFAULT_MILD_PROMPT
            self.summary_template = sum_box.get("1.0", tk.END).strip() or DEFAULT_TRANSLATE_PROMPT
            self._save_cfg()
            dlg.destroy()

        ttk.Button(dlg, text="保存", command=apply).pack(anchor="e", padx=10, pady=(0, 12))

    def _poll_queue(self) -> None:
        try:
            while True:
                bundle = self.out_q.get_nowait()
                self._update_texts(bundle)
        except queue.Empty:
            pass
        self.after(120, self._poll_queue)

    def _update_texts(self, bundle: TextBundle) -> None:
        if not bundle.skipped:
            self._write_text(self.txt_basic, bundle.basic)
            if bundle.mild:
                self._write_text(self.txt_mild, bundle.mild)
            if bundle.summary:
                self._write_text(self.txt_summary, bundle.summary)

        self._accumulate_tokens(bundle)
        self._refresh_token_labels()
        self._log(self._format_log(bundle))

    def _format_log(self, bundle: TextBundle) -> str:
        t = bundle.timings
        info = f"rec={t.get('record_ms', 0):.0f}ms vad(rms={t.get('rms', 0):.1f},vr={t.get('voice_ratio', 0):.2f})"
        if "transcribe_ms" in t:
            info += f" whisper={t['transcribe_ms']:.0f}ms"
        if "gpt_total_ms" in t:
            info += f" gpt_total={t['gpt_total_ms']:.0f}ms (mild={t.get('mild_ms',0):.0f} / sum={t.get('summary_ms',0):.0f})"
        status = "skip:" + bundle.reason if bundle.skipped else "ok"
        preview = bundle.basic[:40].replace("\n", " ")
        return f"[{_now()}] {status} {info} | {preview}"

    def _accumulate_tokens(self, bundle: TextBundle) -> None:
        for k in ("prompt", "completion", "total"):
            self.mild_tokens[k] += bundle.mild_usage.get(k, 0)
            self.summary_tokens[k] += bundle.summary_usage.get(k, 0)

    def _refresh_token_labels(self) -> None:
        self.lbl_mild_tokens.configure(
            text=f"Mild prompt/completion/total: {self.mild_tokens['prompt']} / {self.mild_tokens['completion']} / {self.mild_tokens['total']}"
        )
        self.lbl_summary_tokens.configure(
            text=f"Summary prompt/completion/total: {self.summary_tokens['prompt']} / {self.summary_tokens['completion']} / {self.summary_tokens['total']}"
        )
        self.token_label.configure(
            text=f"mild total: {self.mild_tokens['total']} | summary total: {self.summary_tokens['total']}"
        )

    def _api_status_text(self) -> str:
        if not self.api_key:
            return "APIキー未設定"
        if not SD_AVAILABLE:
            return f"録音不可: {SD_IMPORT_ERROR}"
        if not OpenAI:
            return f"openai import error: {OPENAI_IMPORT_ERROR}"
        return "準備OK"

    def _refresh_status_labels(self) -> None:
        self.api_label.configure(text=self._api_status_text())
        self.model_label.configure(text=f"Whisper: {self.whisper_model} / GPT: {self.gpt_model}")

    def _write_text(self, widget: ScrolledText, text: str) -> None:
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.see(tk.END)
        widget.configure(state=tk.DISABLED)

    def _log(self, text: str) -> None:
        self.txt_log.configure(state=tk.NORMAL)
        self.txt_log.insert(tk.END, text + "\n")
        if float(self.txt_log.index("end-1c").split(".")[0]) > 500:
            self.txt_log.delete("1.0", "220.0")
        self.txt_log.see(tk.END)
        self.txt_log.configure(state=tk.DISABLED)


def _test_filters() -> None:  # pragma: no cover - manual helper
    proc = SpeechProcessor(api_key="dummy", whisper_model="dummy", gpt_model="dummy")  # type: ignore[arg-type]
    assert proc.looks_hallucination("ご視聴ありがとうございました", rms=2.0)
    assert proc.looks_hallucination("高評価お願いします", rms=5.0)
    assert proc.looks_hallucination("あ", rms=1.0)
    assert not proc.looks_hallucination("今日はいい天気ですね", rms=50.0)


def main() -> None:
    app = SubtitleStudio()
    app.mainloop()


if __name__ == "__main__":
    main()
