import tkinter as tk
from tkinter import font
from tkinter import ttk
from PIL import Image, ImageTk
from pathlib import Path
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory

def select_model_file(title, filetypes):
    root = tk.Tk()
    root.withdraw()
    file_path = askopenfilename(
        title=title,
        filetypes=filetypes
    )
    root.destroy()
    return file_path or None
def select_directory(title="Select a directory"):
    root = tk.Tk()
    root.withdraw()
    dir_path = askdirectory(title=title)
    root.destroy()
    return dir_path or None


def option_dialog(title, question, options):
    root = tk.Tk()
    root.withdraw()

    dialog = tk.Toplevel(root)
    dialog.title(title)
    dialog.resizable(False, False)
    dialog.grab_set()

    label = tk.Label(dialog, text=question, pady=10, padx=20, anchor='center', justify='center')
    label.grid(row=0, column=0, columnspan=1, pady=(10, 5), padx=20)

    user_choice = {'value': None}

    def on_click(option):
        user_choice['value'] = option
        dialog.destroy()

    max_btn_width = max(len(opt) for opt in options) + 4

    for i, option in enumerate(options, start=1):
        btn = tk.Button(dialog, text=option, width=max_btn_width, command=lambda opt=i: on_click(opt))
        btn.grid(row=i, column=0, pady=5, padx=20)

    dialog.update_idletasks()
    w = dialog.winfo_width()
    h = dialog.winfo_height()
    ws = dialog.winfo_screenwidth()
    hs = dialog.winfo_screenheight()
    x = (ws // 2) - (w // 2)
    y = (hs // 2) - (h // 2)
    dialog.geometry(f'{w}x{h}+{x}+{y}')

    root.wait_window(dialog)
    root.destroy()
    return user_choice['value']
def show_main_panel(title, question, options, model_path, categories, states):
    root = tk.Tk()
    root.withdraw()

    dialog = tk.Toplevel(root)
    dialog.title(title)
    dialog.resizable(False, False)
    dialog.grab_set()

    bold_font = font.nametofont("TkDefaultFont").copy()
    bold_font.configure(weight="bold")

    label = tk.Label(dialog, text=question, pady=10, padx=20, anchor='center', justify='center', font=bold_font)
    label.grid(row=0, column=0, pady=(10, 5), padx=20)

    user_choice = {'value': None}
    def on_click(option):
        user_choice['value'] = option
        dialog.destroy()

    max_btn_width = max(len(opt) for opt in options) + 4
    for option, state, i in zip(options, states, range(1, len(options) + 1)):
        btn = tk.Button(dialog, text=option, width=max_btn_width, command=lambda opt=i: on_click(opt), state=state)
        btn.grid(row=i, column=0, pady=5, padx=20)

    fixed_info = [{"name": "Model information", "font": "bold"}]
    if model_path is None and categories is None:
        fixed_info.append({"name": "No model selected"})
    if model_path:
        fixed_info.append({"name": "Path: " + model_path, "sticky": "w", "font": "bold"})
    if categories:
        fixed_info.append({"name": "Num categories: " + str(len(categories)), "sticky": "w", "font": "bold"})
        fixed_info.append({"name": "Categories: ", "sticky": "w", "font": "bold"})

    for i, part in enumerate(fixed_info):
        label_font = font.nametofont("TkDefaultFont").copy()
        label_font.configure(weight=part.get("font", "normal"))
        lbl = tk.Label(dialog,
                       text=part["name"],
                       anchor="w",
                       justify="left",
                       font=label_font)
        lbl.grid(column=1, row=i, sticky=part.get("sticky", "w"), pady=2, padx=5)

    scroll_start_row = len(fixed_info)
    total_rows = len(options) + 1

    scroll_frame = tk.Frame(dialog)

    scroll_frame.grid(row=scroll_start_row, column=1, rowspan=total_rows - scroll_start_row + 1,
                      sticky='nsew', padx=5, pady=5)

    scroll_frame.grid_propagate(False)
    scroll_frame.config(width=300, height=150)

    canvas = tk.Canvas(scroll_frame, width=300, height=150)
    scrollbar = tk.Scrollbar(scroll_frame, orient="vertical", command=canvas.yview)
    inner_frame = tk.Frame(canvas)

    inner_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=inner_frame, anchor='nw')
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side='left', fill='both', expand=True)
    scrollbar.pack(side='right', fill='y')

    if categories:
        for i, cat in enumerate(categories):
            lbl = tk.Label(inner_frame, text=cat, anchor="w", justify="left")
            lbl.grid(column=0, row=i, sticky='w', pady=1, padx=5)
    else:
        lbl = tk.Label(inner_frame, text="No categories available", anchor="w", justify="left")
        lbl.grid(column=0, row=0, sticky='w', pady=1, padx=5)

    for r in range(scroll_start_row, total_rows + 1):
        dialog.grid_rowconfigure(r, weight=1)
    dialog.grid_columnconfigure(1, weight=1)

    dialog.update_idletasks()
    w = dialog.winfo_width()
    h = dialog.winfo_height()
    ws = dialog.winfo_screenwidth()
    hs = dialog.winfo_screenheight()
    x = (ws // 2) - (w // 2)
    y = (hs // 2) - (h // 2)
    dialog.geometry(f'{w}x{h}+{x}+{y}')

    root.wait_window(dialog)
    root.destroy()
    return user_choice['value']
def display_confusion_matrix(matrix):
    import NetHandler
    flattened = NetHandler.flatten_confusion_matrix(matrix)
    root = tk.Tk()
    root.withdraw()

    dialog = tk.Toplevel(root)
    dialog.title("Confusion matrix")
    dialog.resizable(False, False)
    dialog.grab_set()

    for i in range(len(flattened)):
        for j in range(len(flattened[i])):
            label = tk.Label(dialog, text=flattened[i][j], pady=10, padx=10, anchor='center', justify='center', bg="green" if i == j and i != 0 else None)
            label.grid(row=i, column=j, pady=10, padx=10)

    root.wait_window(dialog)
    root.destroy()
def show_category_browser(output_root: Path, thumb_size=(128, 128)):
    output_root = Path(output_root)
    supported_exts = (".jpg", ".jpeg", ".png")

    categories = [f for f in output_root.iterdir() if f.is_dir()]

    if not categories:
        print("No categorized folders found.")
        return

    root = tk.Tk()
    root.title("Image Category Browser")
    root.geometry("1000x600")
    root.grab_set()

    left_frame = ttk.Frame(root, width=200)
    left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

    ttk.Label(left_frame, text="Categories:", font=("Arial", 12, "bold")).pack(anchor="w")

    right_frame = ttk.Frame(root)
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(right_frame)
    scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    image_refs = []

    def show_images(category_path: Path):
        nonlocal image_refs
        for widget in scrollable_frame.winfo_children():
            widget.destroy()

        image_refs = []
        image_paths = list(category_path.glob("*"))
        image_paths = [p for p in image_paths if p.suffix.lower() in supported_exts]

        for i, img_path in enumerate(image_paths):
            try:
                img = Image.open(img_path).convert("RGB")
                img.thumbnail(thumb_size)
                img_tk = ImageTk.PhotoImage(img)
                label = ttk.Label(scrollable_frame, image=img_tk, text=img_path.name, compound="top")
                label.grid(row=i // 5, column=i % 5, padx=5, pady=5)
                image_refs.append(img_tk)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")

    for cat in sorted(categories):
        btn = ttk.Button(left_frame, text=cat.name, width=25, command=lambda c=cat: show_images(c))
        btn.pack(anchor="w", pady=2)

    root.mainloop()