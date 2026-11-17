"""Colors and Tk styles for the video review workspace."""

from PIL import Image, ImageDraw, ImageTk

BACKGROUND = "#0e151e"
PANEL = "#17212c"
CANVAS = "#101923"
CONTROL = "#232f3d"
HOVER = "#313e4e"
BORDER = "#303e4d"
TEXT = "#eef4fa"
MUTED = "#a8b9cb"
ACCENT = "#80d9d7"
SELECTED = "#2b3b4b"
DISABLED = "#76899b"
NUMERIC = "Consolas"


def _rounded(master, fill, edge, *, size=20, radius=7, surround=PANEL):
    pixels = size * 4
    canvas = Image.new("RGB", (pixels, pixels), surround)
    ImageDraw.Draw(canvas).rounded_rectangle((1, 1, pixels - 2, pixels - 2), radius=radius * 4,
                                            fill=fill, outline=edge, width=4)
    return ImageTk.PhotoImage(canvas.resize((size, size), Image.Resampling.LANCZOS), master=master)


def _button_images(style):
    assets = {}

    def surface(fill, edge):
        colors = fill, edge
        if colors not in assets:
            assets[colors] = _rounded(style.master, fill, edge, size=64)
        return assets[colors]

    def add_surface(name, fill, hover, pressed, edge=None):
        element = f"Review{name}.surface"
        style.element_create(element, "image", surface(fill, edge),
                             ("disabled", surface(PANEL, None)),
                             ("pressed", "focus", surface(pressed, TEXT)),
                             ("active", "focus", surface(hover, TEXT)),
                             ("focus", surface(fill, TEXT)),
                             ("pressed", surface(pressed, edge)),
                             ("active", surface(hover, edge)),
                             border=8, padding=1, width=20, height=20, sticky="nsew")
        return element

    secondary = add_surface("Secondary", CONTROL, HOVER, SELECTED, "#34485c")
    quiet = add_surface("Quiet", PANEL, CONTROL, SELECTED)
    primary = add_surface("Primary", "#344757", "#41576b", "#2b3b49", "#53687a")

    for style_name, element, foreground in (
        ("TButton", secondary, TEXT),
        ("Accent.TButton", primary, TEXT),
        ("Quiet.TButton", quiet, MUTED),
        ("Confirm.ReviewAction.TButton", add_surface("Confirm", PANEL, "#293e39", "#253531", "#415951"), "#b7cfbf"),
        ("Reject.ReviewAction.TButton", add_surface("Reject", PANEL, "#40333a", "#352b31", "#59434c"), "#d3b5bb"),
        ("Uncertain.ReviewAction.TButton", add_surface("Uncertain", PANEL, "#383548", "#302c3d", "#504a61"), "#c7bdcf"),
    ):
        style.layout(style_name, [(element, {"sticky": "nsew", "children": [
            ("Button.padding", {"sticky": "nsew", "children": [("Button.label", {"sticky": "nsew"})]})
        ]})])
        style.configure(style_name, foreground=foreground)

    style.layout("TMenubutton", [(quiet, {"sticky": "nsew", "children": [
        ("Menubutton.indicator", {"side": "right"}),
        ("Menubutton.padding", {"sticky": "we", "children": [("Menubutton.label", {"side": "left"})]})
    ]})])

    for caption in ("播放", "暂停"):
        canvas = Image.new("RGBA", (56, 56))
        draw = ImageDraw.Draw(canvas)
        if caption == "播放":
            draw.line(((16, 9), (46, 28), (16, 47), (16, 9)), fill=TEXT, width=5, joint="curve")
        else:
            draw.line((18, 10, 18, 46), fill=TEXT, width=5)
            draw.line((38, 10, 38, 46), fill=TEXT, width=5)
        assets[caption] = ImageTk.PhotoImage(canvas.resize((14, 14), Image.Resampling.LANCZOS), master=style.master)
    return assets


def apply(style):
    style.configure(".", foreground=TEXT, background=PANEL, bordercolor=BORDER,
                    darkcolor=BORDER, lightcolor=BORDER, troughcolor=CANVAS, arrowcolor=MUTED,
                    selectbackground=SELECTED, selectforeground=TEXT)
    style.configure("TFrame", background=PANEL)
    style.configure("Workspace.TFrame", background=BACKGROUND)
    style.configure("TLabel", background=PANEL, foreground=TEXT)
    style.configure("Muted.TLabel", foreground=MUTED)
    style.configure("Empty.TLabel", background=CANVAS, foreground=MUTED)
    style.configure("Workspace.TLabel", background=BACKGROUND, foreground=MUTED)
    style.configure("PlayerStatus.TFrame", background=CANVAS)
    style.configure("FrameNumber.TLabel", background=CANVAS, foreground=TEXT, font=(NUMERIC, 13, "bold"))
    style.configure("FrameInfo.TLabel", background=CANVAS, foreground=MUTED, font=(NUMERIC, 10))
    style.configure("VideoMeta.TLabel", foreground=MUTED, font=(NUMERIC, 10))
    style.configure("FrameEvent.TLabel", background=CANVAS, font=("Microsoft YaHei UI", 9))

    style.configure("TButton", background=PANEL, padding=(11, 6), width=0)
    style.map("TButton", background=[], foreground=[("disabled", DISABLED)])
    style.configure("Accent.TButton", background=PANEL, padding=(10, 5), font=("Microsoft YaHei UI", 10, "bold"), space=4)
    style.map("Accent.TButton", background=[], foreground=[("disabled", DISABLED)])
    style.configure("Small.TButton", padding=(8, 4))
    style.configure("Quiet.TButton", padding=(8, 4))
    style.configure("ReviewAction.TButton", padding=(7, 5))
    style.configure("TMenubutton", padding=(8, 5), width=0, foreground=MUTED, arrowsize=8, arrowpadding=(3, 0, 4, 0))
    style.configure("Panel.TMenubutton", padding=(6, 4), font=("Microsoft YaHei UI", 9))
    style.configure("TCheckbutton", foreground=MUTED, background=PANEL, indicatorbackground=CONTROL,
                    indicatorforeground=TEXT, focuscolor=ACCENT)
    style.map("TCheckbutton", indicatorbackground=[("selected", SELECTED)], background=[("active", PANEL)])
    style.configure("TEntry", fieldbackground=CANVAS, foreground=TEXT, insertcolor=TEXT, padding=4)
    style.map("TEntry", bordercolor=[("focus", ACCENT)])
    style.configure("TCombobox", fieldbackground=CONTROL, background=CONTROL, foreground=TEXT, padding=4)
    style.map("TCombobox", fieldbackground=[("readonly", CONTROL)], foreground=[("readonly", TEXT)],
              selectforeground=[("readonly", TEXT)], selectbackground=[("readonly", CONTROL)],
              bordercolor=[("focus", ACCENT)])
    style.configure("Treeview", fieldbackground=CANVAS, background=CANVAS, foreground=TEXT)
    style.configure("Review.Treeview", font=(NUMERIC, 10))
    style.map("Treeview", background=[("selected", SELECTED)], foreground=[("selected", TEXT)])
    style.configure("Treeview.Heading", background=CONTROL, foreground=MUTED)
    style.map("Treeview.Heading", background=[("active", HOVER)])
    style.configure("Vertical.TScrollbar", background=CONTROL, troughcolor=CANVAS, bordercolor=CANVAS, arrowsize=11)
    style.map("Vertical.TScrollbar", background=[("active", HOVER)])
    style.configure("TPanedwindow", background=BACKGROUND, sashthickness=6)
    style.configure("TSeparator", background=BORDER)
    style.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor=CANVAS, thickness=2)
    assets = _button_images(style)
    # Tk tiles the center of bordered images. A wide solid center keeps full-card
    # repaints cheap while the explicit minimum size preserves widget geometry.
    card = _rounded(style.master, PANEL, None, size=256, radius=12, surround=BACKGROUND)
    style.element_create("ReviewCard.surface", "image", card, border=14, width=32, height=32, sticky="nsew")
    style.layout("ReviewCard.TFrame", [("ReviewCard.surface", {"sticky": "nsew"})])
    assets["card"] = card
    return assets
