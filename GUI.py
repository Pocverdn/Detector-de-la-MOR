import tkinter as tk
from tkinter import filedialog
from ctypes import windll

from main import run_pipeline


def start_process(target_label, origin_var, dest_var,logger):
    origin = origin_var.get()
    dest = dest_var.get()

    if not origin or not dest:
        logger("Selecciona ambas carpetas")
        return
    
    target_label.delete("1.0", tk.END)
    logger("Iniciando proceso...")

    run_pipeline(origin, dest, log=logger)

def open_folder(target_label, var):
    file_path = filedialog.askdirectory()
    if file_path:
        target_label.config(text=file_path)
        var.set(file_path)




def gui():

    root = tk.Tk()
    root.title("El detector de la MOR")

    origin_var = tk.StringVar()
    dest_var = tk.StringVar()

    font_big = ("Arial", 13)

    def gui_logger(text):
        logs.insert(tk.END, text + "\n")
        logs.see(tk.END)
        logs.update_idletasks()

    window_width = 800
    window_height = 300

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    center_x = int(screen_width/2 - window_width / 2)
    center_y = int(screen_height/2 - window_height / 2)

    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

    origin_frame = tk.Frame(root)
    origin_frame.pack(pady=5)

    dest_frame = tk.Frame(root)
    dest_frame.pack(pady=5)

    results_frame = tk.Frame(root)
    results_frame.pack(pady=5)

    #ORIGIN

    origin_label = tk.Label(origin_frame, text="Carpeta de origen:", font=font_big)
    origin_url = tk.Label(origin_frame, text="", width=50, relief="solid", font=font_big)
    origin_button = tk.Button(origin_frame, text="Abrir carpeta", command=lambda: open_folder(origin_url, origin_var), font=font_big)

    origin_label.pack(side="left", padx=5)
    origin_url.pack(side="left", padx=5)
    origin_button.pack(side="left", padx=5)

    # DESTINATION

    dest_label = tk.Label(dest_frame, text="Carpeta de destino:", font=font_big)
    dest_url = tk.Label(dest_frame, text="", width=50, relief="solid", font=font_big)
    dest_button = tk.Button(dest_frame, text="Abrir carpeta", command=lambda: open_folder(dest_url, dest_var), font=font_big)

    dest_label.pack(side="left", padx=5)
    dest_url.pack(side="left", padx=5)
    dest_button.pack(side="left", padx=5)


    # RESULTS

    logs = tk.Text(results_frame, width=70, height=10, relief="solid", font=font_big)
    start = tk.Button(results_frame, text="Empezar", command=lambda: start_process(logs, origin_var, dest_var, gui_logger), font=font_big)
    logs.pack(side="left", padx=5)
    start.pack(side="left", padx=5)

    try:
        windll.shcore.SetProcessDpiAwareness(1)
    finally:
        root.mainloop()


if __name__ == "__main__":
    gui()