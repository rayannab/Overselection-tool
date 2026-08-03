import tkinter as tk
import sys
import cv2
import numpy as np
from pathlib import Path
import json
from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).parent))
from masks import feature_mask


DISPLAY_W = 520
DISPLAY_H = 360


def main():
    if len(sys.argv) < 4:
        print("Usage: threshold_tuner.py <excel_file> <video_dir_name> <output_folder>")
        sys.exit(1)

    output_folder = Path(sys.argv[3])

    # Collect all non-processed jpg images recursively
    images = sorted([
        p for p in output_folder.rglob("*.jpg")
        if not p.name.endswith("_processed.jpg")
    ])

    if not images:
        tk.messagebox.showerror("Erreur", "Aucune image trouvée dans le dossier de sortie.")
        sys.exit(0)

    config_path = output_folder / "threshold_config.json"
    default_big = 190
    default_small = 150

    if config_path.exists():
        try:
            with open(config_path) as f:
                cfg = json.load(f)
                default_big = cfg.get("thresh_big", 190)
                default_small = cfg.get("thresh_small", 150)
        except Exception:
            pass

    # ── Root window ──
    root = tk.Tk()
    root.title("Calibration des seuils de détection")
    root.configure(bg="#F5F5F5")
    root.resizable(False, False)

    # ── State ──
    current_idx = [0]
    _photo_refs = [None, None]

    # ── Top bar: navigation ──
    nav_frame = tk.Frame(root, bg="#F5F5F5")
    nav_frame.pack(fill="x", padx=10, pady=(8, 2))

    idx_label = tk.Label(nav_frame, text="", bg="#F5F5F5", font=("Arial", 10, "bold"))
    idx_label.pack(side="left", padx=5)

    tk.Button(nav_frame, text="◀ Précédente", command=lambda: navigate(-1),
              bg="#E0E0E0", relief="flat", padx=8, pady=3).pack(side="left", padx=4)
    tk.Button(nav_frame, text="Suivante ▶", command=lambda: navigate(1),
              bg="#E0E0E0", relief="flat", padx=8, pady=3).pack(side="left", padx=4)

    img_name_label = tk.Label(nav_frame, text="", bg="#F5F5F5", fg="#757575", font=("Arial", 9))
    img_name_label.pack(side="left", padx=10)

    # ── Image panels ──
    panels_frame = tk.Frame(root, bg="#F5F5F5")
    panels_frame.pack(padx=10, pady=5)

    tk.Label(panels_frame, text="Image originale", bg="#F5F5F5",
             font=("Arial", 10, "bold")).grid(row=0, column=0, pady=(0, 4))
    tk.Label(panels_frame, text="Masque binaire (résultat de la détection)",
             bg="#F5F5F5", font=("Arial", 10, "bold")).grid(row=0, column=1, pady=(0, 4))

    canvas_orig = tk.Label(panels_frame, bg="#222222", width=DISPLAY_W, height=DISPLAY_H)
    canvas_orig.grid(row=1, column=0, padx=5)
    canvas_mask = tk.Label(panels_frame, bg="#222222", width=DISPLAY_W, height=DISPLAY_H)
    canvas_mask.grid(row=1, column=1, padx=5)

    # ── Sliders ──
    sliders_frame = tk.Frame(root, bg="#F5F5F5", pady=4)
    sliders_frame.pack(fill="x", padx=20)

    thresh_big_var = tk.IntVar(value=default_big)
    thresh_small_var = tk.IntVar(value=default_small)

    tk.Label(sliders_frame, text="thresh_big (haute luminosité ≥ 100) :",
             bg="#F5F5F5", font=("Arial", 10)).grid(row=0, column=0, sticky="w", pady=3)
    tk.Scale(sliders_frame, from_=0, to=255, orient="horizontal",
             variable=thresh_big_var, length=320, showvalue=False,
             command=lambda _: refresh_mask()).grid(row=0, column=1, padx=8)
    tk.Label(sliders_frame, textvariable=thresh_big_var, bg="#F5F5F5",
             font=("Arial", 10, "bold"), width=4).grid(row=0, column=2)

    tk.Label(sliders_frame, text="thresh_small (basse luminosité < 100) :",
             bg="#F5F5F5", font=("Arial", 10)).grid(row=1, column=0, sticky="w", pady=3)
    tk.Scale(sliders_frame, from_=0, to=255, orient="horizontal",
             variable=thresh_small_var, length=320, showvalue=False,
             command=lambda _: refresh_mask()).grid(row=1, column=1, padx=8)
    tk.Label(sliders_frame, textvariable=thresh_small_var, bg="#F5F5F5",
             font=("Arial", 10, "bold"), width=4).grid(row=1, column=2)

    lum_label = tk.Label(sliders_frame, text="Luminosité : —    Seuil actif : —",
                         bg="#F5F5F5", fg="#555555", font=("Arial", 9, "italic"))
    lum_label.grid(row=2, column=0, columnspan=3, sticky="w", pady=(4, 2))

    # ── Bottom bar ──
    bottom_frame = tk.Frame(root, bg="#EEEEEE", bd=1, relief="sunken")
    bottom_frame.pack(fill="x", padx=0, pady=(8, 0))

    def save_config():
        cfg = {"thresh_big": thresh_big_var.get(), "thresh_small": thresh_small_var.get()}
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        save_btn.config(text="✅ Sauvegardé !", bg="#388E3C")
        root.after(2500, lambda: save_btn.config(text="💾 Sauvegarder", bg="#1976D2"))

    save_btn = tk.Button(bottom_frame, text="💾 Sauvegarder", command=save_config,
                         bg="#1976D2", fg="white", font=("Arial", 10, "bold"),
                         padx=15, pady=6, relief="flat", cursor="hand2")
    save_btn.pack(side="left", padx=10, pady=6)

    tk.Button(bottom_frame, text="Fermer", command=root.destroy,
              bg="#757575", fg="white", font=("Arial", 10),
              padx=15, pady=6, relief="flat", cursor="hand2").pack(side="left", padx=5, pady=6)

    tk.Label(bottom_frame,
             text="Les valeurs sont sauvegardées dans threshold_config.json",
             bg="#EEEEEE", fg="#9E9E9E", font=("Arial", 8)).pack(side="right", padx=10)

    # ── Core logic ──
    def load_cv2_image(path):
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            return None, None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_bgr, img_rgb

    def resize_for_display(img_rgb):
        h, w = img_rgb.shape[:2]
        scale = min(DISPLAY_W / w, DISPLAY_H / h)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img_rgb, (new_w, new_h))

    def refresh_mask():
        idx = current_idx[0]
        _img_bgr, img_rgb = load_cv2_image(images[idx])
        if img_rgb is None:
            return

        # Average luminosity (grayscale mean)
        avg_lum = float(np.array(Image.fromarray(img_rgb).convert("L")).mean())

        # Pick the right threshold based on luminosity
        if avg_lum >= 100:
            thresh = thresh_big_var.get()
            active = f"thresh_big = {thresh}"
        else:
            thresh = thresh_small_var.get()
            active = f"thresh_small = {thresh}"

        mask = feature_mask(img_rgb, thresh)

        lum_label.config(
            text=f"Luminosité moyenne : {avg_lum:.0f}   →   Seuil actif : {active}"
        )

        # Display original
        disp_orig = resize_for_display(img_rgb)
        ph_orig = ImageTk.PhotoImage(image=Image.fromarray(disp_orig))
        canvas_orig.config(image=ph_orig)
        _photo_refs[0] = ph_orig

        # Display mask (white pixels on black background)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        disp_mask = resize_for_display(mask_rgb)
        ph_mask = ImageTk.PhotoImage(image=Image.fromarray(disp_mask))
        canvas_mask.config(image=ph_mask)
        _photo_refs[1] = ph_mask

    def navigate(delta):
        current_idx[0] = (current_idx[0] + delta) % len(images)
        idx = current_idx[0]
        idx_label.config(text=f"Image {idx + 1} / {len(images)}")
        img_name_label.config(text=images[idx].name)
        refresh_mask()

    # Initial render
    navigate(0)

    root.mainloop()


if __name__ == "__main__":
    main()
