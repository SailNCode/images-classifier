import tkinter as tk

def option_dialog(title, question, options):
    root = tk.Tk()
    root.withdraw()  # Hide main window

    # Create a new top-level window for the dialog
    dialog = tk.Toplevel(root)
    dialog.title(title)
    dialog.geometry("300x150")
    dialog.grab_set()  # Modal window

    label = tk.Label(dialog, text=question, pady=20)
    label.pack()

    user_choice = {'value': None}

    def on_click(option):
        user_choice['value'] = option
        dialog.destroy()

    # Create buttons for each option
    for option in options:
        btn = tk.Button(dialog, text=option, width=10, command=lambda opt=option: on_click(opt))
        btn.pack(pady=5)

    root.wait_window(dialog)  # Wait until dialog is closed
    root.destroy()
    return user_choice['value']