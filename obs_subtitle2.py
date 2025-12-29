"""
OBS Subtitle Studio (GUI + engine)
----------------------------------
- GUI: 「開始/停止」でファイル監視＋エンジン起動を統一。処理中/待機中が見える。
- Engine: `python obs_subtitle2.py --engine` でGUIなしエンジン。status.jsonを更新。

status.json (例)
{
  "engine": "running|stopped|error",
  "stage": "listening|transcribing|formatting|writing|idle|error",
  "last_update_ts": 1735600000.0,
  "last_text": "...",
  "latency_ms": {"listen": 20, "transcribe": 800, "format": 300, "write": 50},
  "error": "..."
}

環境変数:
  SUB_JP_RAW / SUB_JP / SUB_JP_TRANS : 出力ファイル
  SUB_STATUS_JSON : ステータスファイル (default C:\\obs\\subtitle_status.json)
  MONITOR_INTERVAL_MS : GUIのポーリング間隔 (default 300)
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import random
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

APP_TITLE = "OBS字幕ビューア"
CONFIG_FILE = "subtitle_gui_config.json"

DEFAULT_STATUS = Path(os.getenv("SUB_STATUS_JSON", r"C:\obs\subtitle_status.json"))


# ---------- common helpers ----------

def _now() -> float:
    return time.time()


def _now_hhmmss() -> str:
    return time.strftime("%H:%M:%S")


def _read_text(path: Path, max_chars: int) -> str:
    try:
        s = path.read_text(encoding="utf-8", errors="replace")
        return s[-max_chars:] if max_chars > 0 and len(s) > max_chars else s
    except FileNotFoundError:
        return ""
    except Exception as e:
        return f"[read error] {e}"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# ---------- engine mode ----------

def write_status(status_path: Path, engine: str, stage: str, last_text: str = "", latency_ms: Dict[str, float] | None = None, error: str = "") -> None:
    payload = {
        "engine": engine,
        "stage": stage,
        "last_update_ts": _now(),
        "last_text": last_text,
        "latency_ms": latency_ms or {},
        "error": error,
    }
    try:
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def engine_loop(status_path: Path, file_raw: Path, file_mild: Path, file_tr: Path) -> None:
    """Simple demo engine: simulates stages and writes status; replace with real logic as needed."""
    try:
        write_status(status_path, "running", "listening")
        while True:
            # simulate listening
            t0 = _now()
            time.sleep(0.3)
            lat = {"listen": (_now() - t0) * 1000}

            # simulate transcribe
            write_status(status_path, "running", "transcribing", latency_ms=lat)
            t1 = _now()
            time.sleep(0.8)
            text = f"sample text {int(t1)%1000}"
            lat["transcribe"] = (_now() - t1) * 1000

            # simulate formatting
            write_status(status_path, "running", "formatting", last_text=text, latency_ms=lat)
            t2 = _now()
            time.sleep(0.3)
            mild = text + " (mild)"
            tr = text + " (summary)"
            lat["format"] = (_now() - t2) * 1000

            # simulate writing
            write_status(status_path, "running", "writing", last_text=tr, latency_ms=lat)
            t3 = _now()
            try:
                file_raw.parent.mkdir(parents=True, exist_ok=True)
                file_raw.write_text(text, encoding="utf-8")
                file_mild.write_text(mild, encoding="utf-8")
                file_tr.write_text(tr, encoding="utf-8")
            except Exception:
                pass
            lat["write"] = (_now() - t3) * 1000

            write_status(status_path, "running", "listening", last_text=tr, latency_ms=lat)
            time.sleep(0.3)
    except KeyboardInterrupt:
        write_status(status_path, "stopped", "idle")
    except Exception as e:
        write_status(status_path, "error", "error", error=str(e))
        raise


# ---------- GUI data ----------

@dataclass
class EngineHandle:
    proc: subprocess.Popen[str] | None = None

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self, timeout: float = 2.0) -> None:
        if not self.is_running():
            return
        assert self.proc
        try:
            self.proc.terminate()
            self.proc.wait(timeout=timeout)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        self.proc = None


# ---------- GUI ----------

class SubtitleGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1250x820")
        self.minsize(1000, 680)

        if load_dotenv:
            try:
                load_dotenv(override=False)
            except Exception:
                pass

        self.config_path = Path(CONFIG_FILE)
        self.cfg = self._load_cfg()

        self.file_raw = Path(os.getenv("SUB_JP_RAW", r"C:\obs\caption_jp_raw.txt"))
        self.file_mild = Path(os.getenv("SUB_JP", r"C:\obs\caption_jp.txt"))
        self.file_tr = Path(os.getenv("SUB_JP_TRANS", r"C:\obs\caption_jp_trans.txt"))
        self.status_path = Path(os.getenv("SUB_STATUS_JSON", DEFAULT_STATUS))

        self.max_chars_var = tk.IntVar(value=int(self.cfg.get("max_chars", 4000)))
        self.refresh_ms_var = tk.IntVar(value=int(os.getenv("MONITOR_INTERVAL_MS", self.cfg.get("refresh_ms", 300))))
        self.font_size_var = tk.IntVar(value=int(self.cfg.get("font_size", 16)))
        self.wrap_var = tk.BooleanVar(value=bool(self.cfg.get("wrap", True)))
        self.topmost_var = tk.BooleanVar(value=bool(self.cfg.get("topmost", False)))

        self.engine_handle = EngineHandle()
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.log_thread: threading.Thread | None = None
        self.log_stop_event = threading.Event()
        self.monitoring = False

        self._build_ui()
        self._apply_font()
        self._apply_topmost()

        self.after(150, self._tick_status)
        self.after(200, self._tick_engine_logs)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ----- config -----
    def _load_cfg(self) -> dict:
        if self.config_path.exists():
            try:
                return json.loads(self.config_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_cfg(self) -> None:
        self.cfg["max_chars"] = int(self.max_chars_var.get())
        self.cfg["refresh_ms"] = int(self.refresh_ms_var.get())
        self.cfg["font_size"] = int(self.font_size_var.get())
        self.cfg["wrap"] = bool(self.wrap_var.get())
        self.cfg["topmost"] = bool(self.topmost_var.get())
        try:
            self.config_path.write_text(json.dumps(self.cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    # ----- UI -----
    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)

        ctrl = ttk.Frame(root)
        ctrl.pack(fill=tk.X, pady=(0, 8))

        self.btn_toggle = ttk.Button(ctrl, text="開始", command=self._toggle_start)
        self.btn_toggle.pack(side=tk.LEFT, padx=(0, 8))

        ttk.Label(ctrl, text="更新(ms)").pack(side=tk.LEFT)
        ttk.Spinbox(ctrl, from_=100, to=3000, increment=50, width=7, textvariable=self.refresh_ms_var,
                    command=self._save_cfg).pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(ctrl, text="表示上限").pack(side=tk.LEFT)
        ttk.Spinbox(ctrl, from_=0, to=50000, increment=500, width=8, textvariable=self.max_chars_var,
                    command=self._save_cfg).pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(ctrl, text="フォント").pack(side=tk.LEFT)
        ttk.Spinbox(ctrl, from_=10, to=36, increment=1, width=5, textvariable=self.font_size_var,
                    command=self._on_font_change).pack(side=tk.LEFT, padx=(4, 12))

        ttk.Checkbutton(ctrl, text="折り返し", variable=self.wrap_var, command=self._on_wrap_toggle).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Checkbutton(ctrl, text="常に手前", variable=self.topmost_var, command=self._on_topmost_toggle).pack(side=tk.LEFT, padx=(0, 12))

        ttk.Button(ctrl, text="ファイル設定", command=self._open_file_settings).pack(side=tk.RIGHT, padx=6)

        status = ttk.Frame(root)
        status.pack(fill=tk.X, pady=(0, 8))
        self.engine_state = ttk.Label(status, text="エンジン：未起動")
        self.engine_state.pack(side=tk.LEFT, padx=(0, 12))
        self.stage_label = ttk.Label(status, text="状態：待機中")
        self.stage_label.pack(side=tk.LEFT, padx=(0, 12))
        self.age_label = ttk.Label(status, text="最終更新：-s")
        self.age_label.pack(side=tk.LEFT, padx=(0, 12))

        self.progress = ttk.Progressbar(status, mode="indeterminate", length=160)
        self.progress.pack(side=tk.LEFT, padx=(8, 0))

        main = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, pady=(4, 8))
        self.txt_raw = self._make_pane(main, "RAW")
        self.txt_mild = self._make_pane(main, "MILD")
        self.txt_tr = self._make_pane(main, "TR")

        ttk.Label(root, text="ログ").pack(anchor="w")
        self.txt_log = ScrolledText(root, height=8, wrap=tk.WORD)
        self.txt_log.pack(fill=tk.BOTH, expand=False)
        self.txt_log.configure(state=tk.DISABLED)

        self.status_bar = ttk.Label(self, text="", anchor="w")
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        self._on_wrap_toggle()

    def _make_pane(self, parent: ttk.Panedwindow, title: str) -> ScrolledText:
        frame = ttk.Frame(parent)
        parent.add(frame, weight=1)
        ttk.Label(frame, text=title).pack(anchor="w")
        txt = ScrolledText(frame, wrap=tk.WORD)
        txt.pack(fill=tk.BOTH, expand=True)
        txt.configure(state=tk.DISABLED)
        return txt

    # ----- actions -----
    def _toggle_start(self) -> None:
        if self.monitoring:
            self._stop_all()
        else:
            self._start_all()

    def _start_all(self) -> None:
        self.monitoring = True
        self.btn_toggle.configure(text="停止")
        self.stage_label.configure(text="状態：起動中")
        self.engine_state.configure(text="エンジン：起動準備")
        self._log("開始を押しました")
        self.progress.start(30)
        self._start_engine()
        self._tick_refresh()

    def _stop_all(self) -> None:
        self.monitoring = False
        self.btn_toggle.configure(text="開始")
        self.progress.stop()
        self.stage_label.configure(text="状態：待機中")
        self.engine_state.configure(text="エンジン：停止中")
        self._stop_engine()
        self._log("停止を押しました")

    def _start_engine(self) -> None:
        if self.engine_handle.is_running():
            self.engine_state.configure(text="エンジン：起動中")
            return
        script = Path(__file__).resolve()
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        cmd = [sys.executable, "-u", str(script), "--engine"]
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(script.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=creationflags,
            )
            self.engine_handle.proc = proc
            self.engine_state.configure(text="エンジン：起動中")
        except Exception as e:
            self.engine_state.configure(text="エンジン：起動失敗")
            self._log(f"エンジン起動失敗: {e}")

    def _stop_engine(self) -> None:
        self.engine_handle.stop()

    def _open_file_settings(self) -> None:
        dlg = tk.Toplevel(self)
        dlg.title("ファイル設定")
        dlg.geometry("760x240")
        dlg.transient(self)
        dlg.grab_set()

        def row(y: int, label: str, initial: Path, setter):
            ttk.Label(dlg, text=label).place(x=10, y=y)
            var = tk.StringVar(value=str(initial))
            ent = ttk.Entry(dlg, width=82, textvariable=var)
            ent.place(x=120, y=y, height=24)

            def browse():
                p = filedialog.askopenfilename(title=label, initialdir=str(initial.parent) if initial.parent.exists() else None)
                if p:
                    var.set(p)
            ttk.Button(dlg, text="参照", command=browse).place(x=680, y=y-1, width=60, height=26)

            def apply():
                setter(Path(var.get()))
            return apply

        apply_raw = row(20, "RAW", self.file_raw, self._set_raw_path)
        apply_mild = row(60, "MILD", self.file_mild, self._set_mild_path)
        apply_tr = row(100, "TR", self.file_tr, self._set_tr_path)
        apply_status = row(140, "status.json", self.status_path, self._set_status_path)

        def ok():
            apply_raw(); apply_mild(); apply_tr(); apply_status()
            dlg.destroy()
        ttk.Button(dlg, text="OK", command=ok).place(x=600, y=180, width=70, height=30)
        ttk.Button(dlg, text="キャンセル", command=dlg.destroy).place(x=680, y=180, width=70, height=30)

    def _set_raw_path(self, p: Path) -> None:
        self.file_raw = p

    def _set_mild_path(self, p: Path) -> None:
        self.file_mild = p

    def _set_tr_path(self, p: Path) -> None:
        self.file_tr = p

    def _set_status_path(self, p: Path) -> None:
        self.status_path = p

    # ----- monitor -----
    def _tick_refresh(self) -> None:
        if not self.monitoring:
            return
        max_chars = int(self.max_chars_var.get())
        for path, widget in (
            (self.file_raw, self.txt_raw),
            (self.file_mild, self.txt_mild),
            (self.file_tr, self.txt_tr),
        ):
            self._write_text(widget, _read_text(path, max_chars))

        status = _read_json(self.status_path)
        self._update_status(status)
        self.after(int(self.refresh_ms_var.get()), self._tick_refresh)

    def _update_status(self, status: dict[str, Any]) -> None:
        stage = status.get("stage", "idle")
        eng = status.get("engine", "unknown")
        last_ts = float(status.get("last_update_ts", 0))
        age = _now() - last_ts if last_ts else 0
        latency = status.get("latency_ms", {})
        self.engine_state.configure(text=f"エンジン：{eng}")
        self.stage_label.configure(text=f"状態：{stage} / latency={latency}")
        self.age_label.configure(text=f"最終更新：{age:.1f}s")
        if stage in ("transcribing", "formatting", "writing"):
            try:
                self.progress.start(30)
            except Exception:
                pass
        else:
            self.progress.stop()
        if status.get("error"):
            self._log(f"エラー: {status.get('error')}")

    def _tick_engine_logs(self) -> None:
        if self.engine_handle.is_running():
            try:
                out = self.engine_handle.proc.stdout  # type: ignore
                if out:
                    chunk = out.readline()
                    if chunk:
                        self._append_log(chunk)
            except Exception:
                pass
        self.after(200, self._tick_engine_logs)

    def _tick_status(self) -> None:
        """定期的にstatus.jsonを読んで表示を更新。例外があってもGUIを落とさない。"""
        try:
            status_path = getattr(self, "status_path", None) or Path(os.getenv("SUB_STATUS_JSON", DEFAULT_STATUS))
            data = _read_json(status_path) if status_path else {}
            stage = data.get("stage", "idle")
            eng = data.get("engine", "not_running")
            last_ts = float(data.get("last_update_ts", 0) or 0)
            age = _now() - last_ts if last_ts else 0.0
            line = f"engine={eng} stage={stage} last={age:.1f}s"

            if hasattr(self, "status_bar"):
                self.status_bar.configure(text=line)
            if hasattr(self, "engine_state"):
                self.engine_state.configure(text=f"エンジン：{eng}")
            if hasattr(self, "stage_label"):
                self.stage_label.configure(text=f"状態：{stage}")
            if hasattr(self, "age_label"):
                self.age_label.configure(text=f"最終更新：{age:.1f}s")

            if hasattr(self, "progress"):
                try:
                    if stage in ("transcribing", "formatting", "writing", "listening"):
                        self.progress.start(30)
                    else:
                        self.progress.stop()
                except Exception:
                    pass
        except Exception as e:
            try:
                self._append_log(f"tick_status error: {e}\n")
            except Exception:
                pass
        finally:
            try:
                self.after(150, self._tick_status)
            except Exception:
                pass

    # ----- misc -----
    def _on_font_change(self) -> None:
        self._apply_font()
        self._save_cfg()

    def _apply_font(self) -> None:
        size = int(self.font_size_var.get())
        font = ("Yu Gothic UI", size) if os.name == "nt" else ("TkDefaultFont", size)
        for t in (self.txt_raw, self.txt_mild, self.txt_tr, self.txt_log):
            try:
                t.configure(font=font)
            except Exception:
                pass

    def _on_wrap_toggle(self) -> None:
        wrap = tk.WORD if self.wrap_var.get() else tk.NONE
        for t in (self.txt_raw, self.txt_mild, self.txt_tr, self.txt_log):
            t.configure(wrap=wrap)
        self._save_cfg()

    def _on_topmost_toggle(self) -> None:
        self._apply_topmost()
        self._save_cfg()

    def _apply_topmost(self) -> None:
        try:
            self.wm_attributes("-topmost", bool(self.topmost_var.get()))
        except Exception:
            pass

    def _on_close(self) -> None:
        self._stop_engine()
        self.destroy()

    # ----- log/text helpers -----
    def _write_text(self, widget: ScrolledText, text: str) -> None:
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.see(tk.END)
        widget.configure(state=tk.DISABLED)

    def _log(self, msg: str) -> None:
        line = f"[{_now_hhmmss()}] {msg}"
        self.status_bar.configure(text=line)
        self._append_log(line + "\n")

    def _append_log(self, text: str) -> None:
        self.txt_log.configure(state=tk.NORMAL)
        self.txt_log.insert(tk.END, text)
        if float(self.txt_log.index("end-1c").split(".")[0]) > 500:
            self.txt_log.delete("1.0", "200.0")
        self.txt_log.see(tk.END)
        self.txt_log.configure(state=tk.DISABLED)


# ---------- main ----------

def gui_main() -> None:
    app = SubtitleGUI()
    app.mainloop()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OBS subtitle GUI/engine")
    p.add_argument("--engine", action="store_true", help="run engine only (no GUI)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.engine:
        status = Path(os.getenv("SUB_STATUS_JSON", DEFAULT_STATUS))
        file_raw = Path(os.getenv("SUB_JP_RAW", r"C:\obs\caption_jp_raw.txt"))
        file_mild = Path(os.getenv("SUB_JP", r"C:\obs\caption_jp.txt"))
        file_tr = Path(os.getenv("SUB_JP_TRANS", r"C:\obs\caption_jp_trans.txt"))
        engine_loop(status, file_raw, file_mild, file_tr)
    else:
        gui_main()


if __name__ == "__main__":
    main()
