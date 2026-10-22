"""Small dialogs without an extra GUI dependency on Windows."""
from pathlib import Path
import json
import os
import subprocess


def ask(title, prompt, *, integer=False, initial="", minimum=0, maximum=0,
        confirm=False, smoke_test=False, kind=None, file_filter='All files|*.*'):
    if os.name == "nt":
        script = Path(__file__).with_name("annotation_dialog.ps1")
        command = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
                   "-Kind", kind or ("confirm" if confirm else "integer" if integer else "text"),
                   "-DialogTitle", title, "-Prompt", prompt, "-Initial", str(initial),
                   "-Minimum", str(minimum), "-Maximum", str(maximum)]
        if kind == 'file':
            command += ['-FileFilter', file_filter]
        if smoke_test:
            command.append("-SmokeTest")
        result = subprocess.run(command, capture_output=True, encoding="utf-8",
                                creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode:
            raise ValueError("系统对话框无法启动：" + result.stderr[-400:])
        response = json.loads(result.stdout.strip().lstrip("\ufeff"))
        return response["value"] if response["ok"] else None
    import tkinter as tk
    from tkinter import simpledialog, messagebox
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        if kind in ('file', 'folder'):
            from tkinter import filedialog
            return (filedialog.askdirectory(title=title, initialdir=initial or None, parent=root)
                    if kind == 'folder' else filedialog.askopenfilename(title=title, initialdir=initial or None, parent=root)) or None
        if confirm:
            return messagebox.askyesno(title, prompt, parent=root)
        if integer:
            return simpledialog.askinteger(title, prompt, parent=root, minvalue=minimum, maxvalue=maximum)
        return simpledialog.askstring(title, prompt, parent=root, initialvalue=initial)
    finally:
        root.destroy()


def pick_file(title, *, initial='', file_filter='All files|*.*'):
    return ask(title, '', kind='file', initial=initial, file_filter=file_filter)


def pick_folder(title, *, initial=''):
    return ask(title, '', kind='folder', initial=initial)
