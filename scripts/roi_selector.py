import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageTk, ImageDraw
import re
import pandas as pd
import sys


class ROISelectorGUI:
    def __init__(self, output_folder="output"):
        self.output_folder = Path(output_folder)
        self.road_shapes = ['straight', 'slight_curve']
        self.excel_sheet =  sys.argv[1]
        self.video_folder = sys.argv[2]
        self.selected_rois = {}
        self.current_shape_idx = 0
        self.current_image_idx = 0
        self.current_ego_idx = 0
        self.points = []
        self.images_by_shape = defaultdict(lambda: defaultdict(list))
        self.ego_ids = []
        self.available_shapes = []
        self.ego_id = None

        self.load_images()

        # GUI
        self.root = tk.Tk()
        self.root.title("ROI Selector - Draw Trapezoid")
        self.root.geometry("1400x900")

        self.setup_ui()

    # ============================================================
    # Load images grouped by road shape
    # ============================================================
    def get_info_from_excel(self, excel_file, subfolder=None):
        """
        Compare the detected lane lines to the ground truth data from Texcel.
        :param: texcel_data The ground truth data from Texcel
        :return: Comparison results
        """
        # This function would contain code to compare the detected lane lines
        # to the ground truth data from Texcel and return the results of the comparison.   
        df = pd.read_excel(excel_file)
        
        pattern = r"event_(\d{8})_(\d{6})_instant(\d+\.?\d*)"
        # Convert Path object to string
        subfolder_str = str(subfolder.name) if isinstance(subfolder, Path) else str(subfolder)
        match = re.search(pattern, subfolder_str)

        if not match:
            raise ValueError(f"Format invalide (interface ROI): {subfolder_str}. Le format attendu est 'event_YYYYMMDD_HHMMSS_instantX.XX'.")

        date_str = match.group(1)
        time_str = match.group(2)
        instant = float(match.group(3))
        for index, row in df.iterrows():
            try :
                instant_raw = row['Debut_event_en_DareDeevil_approx']
            
                if (instant == instant_raw) and (f"{date_str}_{time_str}" in row['Nom_du_log']):
                    ego_id = row['Nom_du_log'].split('_')[2]
                    return ego_id
            except Exception as e:
                print(f"Error processing row {index}: {e}")
                continue
    def load_images(self):
        for ts_folder in self.output_folder.glob("*"):
            if not ts_folder.is_dir():
                continue

            for event_folder in ts_folder.glob("*"):
                if not event_folder.is_dir():
                    continue
                ego_id = self.get_info_from_excel(self.excel_sheet, event_folder)
                if ego_id is None:
                    continue
                    
                for img_path in event_folder.glob("*.jpg"):
                    if img_path.name.endswith("_processed.jpg") :
                        continue

                    filename = img_path.name.lower()
                    for shape in self.road_shapes:
                        if shape in filename:
                            self.images_by_shape[ego_id][shape].append(img_path)
                            break

        # Extraire les ego_ids uniques
        self.ego_ids = list(self.images_by_shape.keys())
        
        # Initialiser avec le premier ego_id
        if self.ego_ids:
            current_ego_id = self.ego_ids[self.current_ego_idx]
            self.available_shapes = [
                shape for shape in self.road_shapes if len(self.images_by_shape[current_ego_id][shape]) > 0
            ]
        else:
            self.available_shapes = []

    # ============================================================
    # GUI Setup
    # ============================================================
    def setup_ui(self):
        top_frame = ttk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.shape_label = ttk.Label(top_frame, text="", font=("Arial", 14, "bold"))
        self.shape_label.pack(side=tk.LEFT, padx=10)

        self.ego_label = ttk.Label(top_frame, text="", font=("Arial", 12, "italic"), foreground="#1976D2")
        self.ego_label.pack(side=tk.LEFT, padx=10)

        self.image_label = ttk.Label(top_frame, text="", font=("Arial", 12))
        self.image_label.pack(side=tk.LEFT, padx=10)

        self.points_label = ttk.Label(top_frame, text="Points: 0/4", font=("Arial", 12))
        self.points_label.pack(side=tk.LEFT, padx=10)

        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg="black", cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        ttk.Button(control_frame, text="Reset Points", command=self.reset_points)\
            .pack(fill=tk.X, pady=5)

        ttk.Button(control_frame, text="← Previous Image", command=self.previous_image)\
            .pack(fill=tk.X, pady=5)

        ttk.Button(control_frame, text="Next Image →", command=self.next_image)\
            .pack(fill=tk.X, pady=5)

        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        self.confirm_btn = ttk.Button(
            control_frame,
            text="✓ Confirm & Next Shape",
            command=self.confirm_roi,
            state=tk.DISABLED
        )
        self.confirm_btn.pack(fill=tk.X, pady=5, ipady=10)

        ttk.Button(control_frame, text="Quit", command=self.quit_app)\
            .pack(fill=tk.X, pady=10)

        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.display_current_image()

    # ============================================================
    # Image display
    # ============================================================
    def display_current_image(self):
        # Vérifier si tous les ego_ids ont été traités
        if self.current_ego_idx >= len(self.ego_ids):
            messagebox.showinfo("Done", "All ROIs selected for all ego_ids.")
            self.save_rois()
            self.root.quit()
            return
        
        # Vérifier si tous les shapes de l'ego_id courant ont été traités
        if self.current_shape_idx >= len(self.available_shapes):
            # Passer à l'ego_id suivant
            self.current_ego_idx += 1
            self.current_shape_idx = 0
            self.current_image_idx = 0
            
            if self.current_ego_idx >= len(self.ego_ids):
                messagebox.showinfo("Done", "All ROIs selected for all ego_ids.")
                self.save_rois()
                self.root.quit()
                return
            
            # Mettre à jour available_shapes pour le nouvel ego_id
            current_ego_id = self.ego_ids[self.current_ego_idx]
            self.available_shapes = [
                shape for shape in self.road_shapes if len(self.images_by_shape[current_ego_id][shape]) > 0
            ]
        
        current_ego_id = self.ego_ids[self.current_ego_idx]
        shape = self.available_shapes[self.current_shape_idx]
        img_list = self.images_by_shape[current_ego_id][shape]

        if self.current_image_idx >= len(img_list):
            self.current_image_idx = 0

        img_path = img_list[self.current_image_idx]

        img = cv2.imread(str(img_path))
        if img is None:
            return

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 10: cw, ch = 1000, 700

        scale = min(cw / w, ch / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        img_resized = cv2.resize(img, (new_w, new_h))

        self.original_shape = (w, h)
        self.display_shape = (new_w, new_h)
        self.current_image = Image.fromarray(img_resized)
        self.current_image_path = img_path

        self.points = []
        self.confirm_btn.config(state=tk.DISABLED)
        self.update_points_label()

        self.redraw_canvas()

        self.shape_label.config(text=f"Road Shape: {shape.upper()}")
        self.ego_label.config(text=f"Ego ID: {current_ego_id}")
        self.image_label.config(text=f"{img_path.name}")
        self.status_var.set("Select 4 points...")

    def redraw_canvas(self):
        img_draw = self.current_image.copy()
        draw = ImageDraw.Draw(img_draw)

        # Draw points
        for i, (x, y) in enumerate(self.points):
            r = 6
            draw.ellipse([x-r, y-r, x+r, y+r], fill="lime")
            draw.text((x+10, y-10), str(i+1), fill="white")

        # Draw trapezoid when ready
        if len(self.points) == 4:
            draw.line([self.points[0], self.points[1]], fill="lime", width=2)
            draw.line([self.points[1], self.points[2]], fill="lime", width=2)
            draw.line([self.points[2], self.points[3]], fill="lime", width=2)
            draw.line([self.points[3], self.points[0]], fill="lime", width=2)

        self.canvas_img = ImageTk.PhotoImage(img_draw)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.canvas_img)

    # ============================================================
    # Canvas click
    # ============================================================
    def on_canvas_click(self, event):
        if len(self.points) >= 4:
            return

        self.points.append((event.x, event.y))
        self.update_points_label()
        self.redraw_canvas()

        if len(self.points) == 4:
            self.confirm_btn.config(state=tk.NORMAL)
            self.status_var.set("4 points selected — click Confirm.")

    def update_points_label(self):
        self.points_label.config(text=f"Points: {len(self.points)}/4")

    # ============================================================
    # Navigation
    # ============================================================
    def reset_points(self):
        self.points = []
        self.confirm_btn.config(state=tk.DISABLED)
        self.redraw_canvas()

    def previous_image(self):
        if self.current_image_idx > 0:
            self.current_image_idx -= 1
            self.display_current_image()

    def next_image(self):
        current_ego_id = self.ego_ids[self.current_ego_idx]
        shape = self.available_shapes[self.current_shape_idx]
        if self.current_image_idx < len(self.images_by_shape[current_ego_id][shape]) - 1:
            self.current_image_idx += 1
            self.display_current_image()

    # ============================================================
    # CONFIRM
    # ============================================================
    def confirm_roi(self):
        current_ego_id = self.ego_ids[self.current_ego_idx]
        shape = self.available_shapes[self.current_shape_idx]

        if len(self.points) != 4:
            return

        scale_x = self.original_shape[0] / self.display_shape[0]
        scale_y = self.original_shape[1] / self.display_shape[1]

        scaled = [(int(x * scale_x), int(y * scale_y)) for x, y in self.points]

        # Créer l'entrée ego_id si elle n'existe pas
        if current_ego_id not in self.selected_rois:
            self.selected_rois[current_ego_id] = {}
        
        self.selected_rois[current_ego_id][shape] = {
            "points": scaled,
            "image": str(self.current_image_path)
        }

        self.current_shape_idx += 1
        self.current_image_idx = 0

        self.display_current_image()

    # ============================================================
    # SAVE JSON
    # ============================================================
    def save_rois(self):
        with open(self.output_folder/f"selected_rois_{self.video_folder}.json", "w", encoding="utf-8") as f:
            json.dump(self.selected_rois, f, indent=4)

        print(f"Saved selected_rois_{self.video_folder}.json.")

    def quit_app(self):
        self.root.quit()

    # ============================================================
    # TUTORIAL POPUP
    # ============================================================
    def show_tutorial_popup(self):
        popup = tk.Toplevel(self.root)
        popup.title("📖 Guide – Configuration des ROIs")
        popup.resizable(False, False)
        popup.grab_set()

        popup.update_idletasks()
        w, h = 520, 360
        x = (popup.winfo_screenwidth() - w) // 2
        y = (popup.winfo_screenheight() - h) // 2
        popup.geometry(f"{w}x{h}+{x}+{y}")

        tk.Label(
            popup,
            text="Comment configurer une ROI ?",
            font=("Arial", 13, "bold")
        ).pack(pady=(18, 10))

        steps = [
            "1.  Naviguez vers une image représentative de la forme de route.",
            "2.  Cliquez sur les 4 coins du trapèze qui délimite la voie :",
            "     haut-gauche  →  bas-gauche  →  bas-droite  →  haut-droite.",
            "3.  Vérifiez que la zone englobe bien les deux lignes de voie.",
            "4.  Cliquez sur  ✓ Confirm & Next Shape  pour valider.",
            "5.  Répétez pour chaque forme de route disponible.",
        ]

        frame_steps = tk.Frame(popup, padx=30)
        frame_steps.pack(fill=tk.X)
        for step in steps:
            tk.Label(frame_steps, text=step, font=("Arial", 10), anchor="w", justify="left").pack(fill=tk.X, pady=1)

        ttk.Separator(popup, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=20, pady=16)

        tk.Button(
            popup,
            text="▶  Commencer",
            command=popup.destroy,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 11, "bold"),
            padx=20,
            pady=6
        ).pack(side=tk.LEFT, padx=(60, 10), pady=(0, 16))

        tk.Button(
            popup,
            text="Passer",
            command=popup.destroy,
            font=("Arial", 10),
            padx=10
        ).pack(side=tk.LEFT, pady=(0, 16))

    def run(self):
        self.root.after(0, self.show_tutorial_popup)
        self.root.mainloop()


def main():
    output_folder = sys.argv[3]
    output_folder_name = output_folder
    app = ROISelectorGUI(output_folder_name)
    app.run()


if __name__ == "__main__":
    main()