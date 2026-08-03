import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import subprocess
import sys
import threading
import os
import webbrowser
import urllib.parse
from frozen_utils import build_subprocess_cmd

def main():
    global process
    process = None
    global script_running
    script_running = False

    def select_excel():
        excel_var.set(filedialog.askopenfilename(
            title="Sélectionner le fichier Excel",
            filetypes=[("Excel files", "*.xlsx *.xls")]
        ))

    def select_video_folder():
        video_var.set(filedialog.askdirectory(title="Sélectionner le dossier vidéo"))

    def select_output_folder():
        output_var.set(filedialog.askdirectory(title="Sélectionner le dossier de sortie"))

    def stop_script():
        global process, script_running
        if process and script_running:
            output_text.insert(tk.END, "\n⛔ Arrêt du script demandé...\n")
            output_text.see(tk.END)
            try:
                process.terminate()
            except:
                pass
            process = None
            script_running = False
            progress_bar.stop()
        else:
            messagebox.showinfo("Info", "Aucun script en cours d'exécution.")


    def run_script():
        global process, script_running

        excel = excel_var.get()
        video = video_var.get()
        output = output_var.get()

        
        selected_categories = []

        if cat1_var.get():
            selected_categories.append("1")
        if cat2_var.get():
            selected_categories.append("2")
        if cat3_var.get():
            selected_categories.append("3")
        if cat4_var.get():
            selected_categories.append("4")
        if not selected_categories:
            messagebox.showerror("Erreur", "Merci de sélectionner au moins une catégorie à analyser.")
            return
        if script_running:
            messagebox.showwarning("Déjà en cours", "Le script est déjà en cours d'exécution.")
            return

        if not excel or not video or not output:
            messagebox.showerror("Erreur", "Merci de remplir tous les champs.")
            return

        output_text.insert(tk.END, "🚀 Démarrage du script...\n")
        output_text.see(tk.END)

        script_running = True
        progress_bar.start(15)

        def task():
            global process, script_running
            
            cmd = build_subprocess_cmd(
                "main_algo.py",
                excel,
                video,
                output,
                "--categories",
                ",".join(selected_categories)
            )

            if create_pdf_var.get():
                cmd.append("--create-pdf")
            if configure_ROIs.get():
                cmd.append("--configure-rois")
            focal_value = focal_length_var.get().strip()
            print("FOCALE DEBUG =", repr(focal_length_var.get()))

            if focal_value != "":
                cmd.append("--focal")
                cmd.append(focal_value)

            cmd.append("--trust-scores")
            cmd.append(f"{trust_cat1_var.get()},{trust_cat2_var.get()},{trust_cat3_var.get()},{trust_cat4_var.get()}")

            lane_width_value = lane_width_var.get().strip()
            if lane_width_value != "":
                cmd.append("--lane-width")
                cmd.append(lane_width_value)

            max_error_value = max_error_var.get().strip()
            if max_error_value != "":
                cmd.append("--max-error")
                cmd.append(max_error_value)

            if calibrate_thresh_var.get():
                cmd.append("--calibrate-thresholds")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='ignore'
            )
            '''''
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "main.py",
                    excel,
                    video,
                    output,
                    "--categories",
                    ",".join(selected_categories)
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='ignore'
            )
                '''''
            for line in process.stdout:
                output_text.insert(tk.END, line)
                output_text.see(tk.END)

            for line in process.stderr:
                output_text.insert(tk.END, "⚠️ ERREUR: " + line)
                output_text.see(tk.END)

            process.wait()
            script_running = False
            progress_bar.stop()

            if process.returncode == 0:
                output_text.insert(tk.END, "\n✅ Analyse terminée avec succès.\n")
                messagebox.showinfo("Terminé", "Analyse terminée avec succès.")
            else:
                output_text.insert(tk.END, "\n❌ Le script a échoué ou a été stoppé.\n")
                messagebox.showerror("Erreur", "Le script a échoué ou a été stoppé.")

        threading.Thread(target=task, daemon=True).start()

    def clear_output():
        output_text.delete("1.0", tk.END)

    def contact_by_mail():
        to = "rayan.naboulsi@ampere.cars"   # 👉 remplace par ton mail
        subject = "Question - Analyse Overselection"
        body = (
            "Bonjour,\n\n"
            "J'ai une question concernant l'application d'analyse.\n\n"
            "Description de mon problème :\n"
            "- \n\n"
            "Cordialement,"
        )

        params = {
            "subject": subject,
            "body": body
        }

        url = f"mailto:{to}?" + urllib.parse.urlencode(params)
        webbrowser.open(url)



    # ============================================
    #   INTERFACE TKINTER AVEC ONGLETS
    # ============================================

    def make_help_btn(parent, msg):
        """Crée un bouton '?' qui affiche un popup d'aide."""
        def show():
            popup = tk.Toplevel(root)
            popup.title("Aide")
            popup.resizable(False, False)
            popup.grab_set()
            popup.update_idletasks()
            w, h = 400, 180
            x = (popup.winfo_screenwidth() - w) // 2
            y = (popup.winfo_screenheight() - h) // 2
            popup.geometry(f"{w}x{h}+{x}+{y}")
            tk.Label(
                popup, text=msg,
                wraplength=365, justify="left",
                font=("Arial", 10), padx=15, pady=12
            ).pack(expand=True, fill="both")
            tk.Button(popup, text="OK", command=popup.destroy, width=8).pack(pady=(0, 10))
        btn = tk.Button(
            parent, text="?", width=2,
            font=("Arial", 9, "bold"), fg="#1976D2",
            relief="flat", cursor="hand2", command=show
        )
        return btn

    def section_header(parent, text, help_msg, pady=(20, 5)):
        """Label de section gras avec bouton '?' aligné."""
        row = tk.Frame(parent)
        row.pack(anchor="w", padx=20, pady=pady)
        tk.Label(row, text=text, font=("Arial", 12, "bold")).pack(side="left")
        make_help_btn(row, help_msg).pack(side="left", padx=(6, 0))

    root = tk.Tk()
    root.title("Analyse Overselection – Interface Locale")
    root.geometry("950x700")

    # ✅ Création du Notebook (les onglets)
    notebook = ttk.Notebook(root)
    frame_main = ttk.Frame(notebook)
    _frame_advanced_outer = ttk.Frame(notebook)

    notebook.add(frame_main, text="Analyse")
    notebook.add(_frame_advanced_outer, text="Paramètres avancés")
    notebook.pack(expand=True, fill="both")

    # Scrollable frame_advanced
    _adv_canvas = tk.Canvas(_frame_advanced_outer, borderwidth=0)
    _adv_scrollbar = ttk.Scrollbar(_frame_advanced_outer, orient="vertical", command=_adv_canvas.yview)
    frame_advanced = ttk.Frame(_adv_canvas)
    frame_advanced.bind(
        "<Configure>",
        lambda e: _adv_canvas.configure(scrollregion=_adv_canvas.bbox("all"))
    )
    _adv_canvas.create_window((0, 0), window=frame_advanced, anchor="nw")
    _adv_canvas.configure(yscrollcommand=_adv_scrollbar.set)
    _adv_canvas.pack(side="left", fill="both", expand=True)
    _adv_scrollbar.pack(side="right", fill="y")
    _frame_advanced_outer.bind(
        "<Enter>",
        lambda e: _adv_canvas.bind_all("<MouseWheel>", lambda ev: _adv_canvas.yview_scroll(int(-1*(ev.delta/120)), "units"))
    )
    _frame_advanced_outer.bind(
        "<Leave>",
        lambda e: _adv_canvas.unbind_all("<MouseWheel>")
    )


    # =====================================================
    # ✅ Onglet 1 : Interface principale (Analyse)
    # =====================================================

    excel_var = tk.StringVar()
    video_var = tk.StringVar()
    output_var = tk.StringVar()

    tk.Label(frame_main, text="Fichier Excel :").pack(anchor="w", padx=10)
    tk.Entry(frame_main, textvariable=excel_var, width=70).pack(padx=10)
    tk.Button(frame_main, text="Parcourir", command=select_excel).pack(pady=5)

    tk.Label(frame_main, text="Dossier Vidéo :").pack(anchor="w", padx=10)
    tk.Entry(frame_main, textvariable=video_var, width=70).pack(padx=10)
    tk.Button(frame_main, text="Parcourir", command=select_video_folder).pack(pady=5)

    tk.Label(frame_main, text="Dossier de sortie :").pack(anchor="w", padx=10)
    tk.Entry(frame_main, textvariable=output_var, width=70).pack(padx=10)
    tk.Button(frame_main, text="Parcourir", command=select_output_folder).pack(pady=5)

    frame_buttons = tk.Frame(frame_main)
    frame_buttons.pack(pady=15)

    tk.Button(frame_buttons, text="▶ Lancer l'analyse", command=run_script,
            bg="#4CAF50", fg="white").pack(side="left", padx=10)

    tk.Button(frame_buttons, text="⛔ Stopper le script", command=stop_script,
            bg="#C62828", fg="white").pack(side="left", padx=10)

    tk.Button(frame_buttons,text="🧹Clear Terminal",command=clear_output,bg="#424242",fg="white").pack(side="left", padx=10)

    progress_bar = ttk.Progressbar(frame_main, mode="indeterminate", length=400)
    progress_bar.pack(pady=10)

    output_text = tk.Text(frame_main, height=25, width=110, bg="black", fg="white")
    output_text.pack(padx=10, pady=10)


    # =====================================================
    # ✅ Onglet 2 : Paramètres avancés 
    # =====================================================

    tk.Label(frame_advanced, text="🔧 Paramètres avancés", font=("Arial", 14, "bold")).pack(pady=10)


    ### CAT A ANALYSER ###
    section_header(
        frame_advanced,
        "Catégories à analyser",
        "CAT1 : le plus critique, TTC < 5s.\n"
        "CAT2 : TTC entre 5 et 10s.\n"
        "CAT3 : event duration > 100 ms, TTC n/a.\n"
        "CAT4 : pas critique, event duration < 100 ms.\n\n"
        "Sélectionnez uniquement les catégories que vous souhaitez analyser.",
        pady=(15, 5)
    )

    categories_frame = tk.Frame(frame_advanced)
    categories_frame.pack(anchor="w", padx=30, pady=5)

    cat1_var = tk.BooleanVar(value=True)
    cat2_var = tk.BooleanVar(value=True)
    cat3_var = tk.BooleanVar(value=True)
    cat4_var = tk.BooleanVar(value=True)
    create_pdf_var = tk.BooleanVar(value=False)
    configure_ROIs = tk.BooleanVar(value=False)
    calibrate_thresh_var = tk.BooleanVar(value=False)
    focal_length_var = tk.StringVar(value="1300")
    lane_width_var = tk.StringVar(value="450")
    max_error_var = tk.StringVar(value="17")

    trust_cat1_var = tk.StringVar(value="80")
    trust_cat2_var = tk.StringVar(value="70")
    trust_cat3_var = tk.StringVar(value="60")
    trust_cat4_var = tk.StringVar(value="50")

    for _cat_label, _cat_var, _trust_var in [
        ("CAT1", cat1_var, trust_cat1_var),
        ("CAT2", cat2_var, trust_cat2_var),
        ("CAT3", cat3_var, trust_cat3_var),
        ("CAT4", cat4_var, trust_cat4_var),
    ]:
        _col = tk.Frame(categories_frame)
        _col.pack(side="left", padx=15)
        tk.Checkbutton(_col, text=_cat_label, variable=_cat_var).pack(anchor="w")
        _tr = tk.Frame(_col)
        _tr.pack(anchor="w")
        tk.Label(_tr, text="Trust :", font=("Arial", 8), fg="gray").pack(side="left")
        tk.Entry(_tr, textvariable=_trust_var, width=5).pack(side="left", padx=2)

    tk.Label(
        frame_advanced,
        text="© 2026 - Application Analyse Overselection",
        fg="gray"
    ).pack(side="bottom", anchor="center", pady=30)

    section_header(
        frame_advanced,
        "Sorties & rapports",
        "Rapport PDF : génère un résumé PDF des résultats d’analyse à la fin du traitement.\n"
        "Utile pour archiver ou partager les résultats sans ouvrir le fichier Excel."
    )

    tk.Checkbutton(
        frame_advanced,
        text="📄 Créer un rapport PDF à la fin de l’analyse",
        variable=create_pdf_var
    ).pack(anchor="w", padx=30, pady=5)

    section_header(
        frame_advanced,
        "Configuration des ROIs",
        "ROI (Region Of Interest) : zone trapézoïdale définie manuellement sur l’image,\n"
        "qui délimite la voie de circulation pour la détection.\n\n"
        "Cocher cette case pour relancer l’éditeur de ROI même si un fichier existe déjà.\n"
        "Recommandé lors d’un changement de caméra ou de véhicule."
    )


    tk.Checkbutton(
        frame_advanced,
        text="Configurer les ROIs",
        variable=configure_ROIs
    ).pack(anchor="w", padx=30, pady=5)

    section_header(
        frame_advanced,
        "Calibration des seuils de détection",
        "En conditions difficiles (nuit, pluie, contre-jour), les seuils par défaut\n"
        "thresh_big=190 et thresh_small=150 peuvent mal détecter les lignes de voie.\n\n"
        "Cocher cette case lance un outil interactif AVANT l'analyse :\n"
        "  • Panneau gauche : image originale\n"
        "  • Panneau droit : masque binaire en temps réel\n"
        "  • 2 sliders pour ajuster thresh_big et thresh_small\n\n"
        "Les valeurs sont sauvegardées dans threshold_config.json\n"
        "et utilisées automatiquement pour toute l'analyse."
    )

    tk.Checkbutton(
        frame_advanced,
        text="Calibrer les seuils de détection (conditions difficiles)",
        variable=calibrate_thresh_var
    ).pack(anchor="w", padx=30, pady=5)

    section_header(
        frame_advanced,
        "Paramètres caméra",
        "Focale (pixels) : distance focale de la caméra en pixels.\n"
        "Elle sert au calcul de la distance longi/lat des targets.\n"
        "Valeurs typiques : 900-1500 px selon la caméra.\n"
        "Laisser vide pour utiliser la détection automatique.\n\n"
        "Largeur de voie (px) : largeur estimée d’une voie dans l’image.\n"
        "Utilisée quand une seule ligne de voie est détectée (l’autre est reconstruite).\n"
        "Valeurs typiques : 400–600 px. Défaut : 450 px."
    )

    tk.Label(
        frame_advanced,
        text="Focale manuelle (optionnelle)",
    ).pack(anchor="w", padx=30)

    focal_entry = tk.Entry(
        frame_advanced,
        textvariable=focal_length_var,
        width=20
    )
    focal_entry.pack(anchor="w", padx=30, pady=5)

    tk.Label(
        frame_advanced,
        text="Laisser vide pour utiliser la focale automatique",
        fg="gray"
    ).pack(anchor="w", padx=30)

    tk.Label(
        frame_advanced,
        text="Largeur de voie (pixels)",
    ).pack(anchor="w", padx=30, pady=(10, 0))
    tk.Entry(
        frame_advanced,
        textvariable=lane_width_var,
        width=20
    ).pack(anchor="w", padx=30, pady=5)
    tk.Label(
        frame_advanced,
        text="Valeur par défaut : 450 px. Augmenter si la voie paraît trop étroite.",
        fg="gray"
    ).pack(anchor="w", padx=30)

    section_header(
        frame_advanced,
        "Matching YOLO ↔ DAREDEEVIL",
        "Erreur max de matching : seuil en dessous duquel une détection YOLO est associée\n"
        "à l’événement du fichier Excel.\n"
        "L’erreur combine l’écart longitudinal et latéral entre la position YOLO\n"
        "et la position indiquée par DAREDEEVIL.\n\n"
        "Défaut : 17. Réduire (ex. 10) pour un matching plus strict,\n"
        "augmenter (ex. 25) si des véhicules valides ne sont pas appariés."
    )
    tk.Label(
        frame_advanced,
        text="Erreur max de matching (erreur longi + lat).",
    ).pack(anchor="w", padx=30)
    tk.Entry(
        frame_advanced,
        textvariable=max_error_var,
        width=20
    ).pack(anchor="w", padx=30, pady=5)
    tk.Label(
        frame_advanced,
        text="Valeur par défaut : 17 . Réduire pour un matching plus strict.",
        fg="gray"
    ).pack(anchor="w", padx=30)

    ############"##################
    # CONTACT PAR MAIL
    ############################### 
    frame_contact = ttk.Frame(notebook)
    notebook.add(frame_contact, text="Nous contacter")
    tk.Label(
        frame_contact,
        text="📬 Nous contacter",
        font=("Arial", 16, "bold")
    ).pack(pady=20)

    tk.Label(
        frame_contact,
        text=(
            "Une question, un problème ou une suggestion ?\n\n"
            "N'hésitez pas à nous contacter par mail.\n"
            "Une réponse vous sera envoyée dans les plus brefs délais."
        ),
        font=("Arial", 11),
        justify="center"
    ).pack(pady=10)

    tk.Button(
        frame_contact,
        text="📧 Contacter par mail",
        command=contact_by_mail,
        bg="#1976D2",
        fg="white",
        font=("Arial", 11, "bold"),
        padx=20,
        pady=10
    ).pack(pady=20)

    tk.Label(
        frame_contact,
        text="© 2026 - Application Analyse Overselection",
        fg="gray"
    ).pack(side="bottom", pady=300)


    root.mainloop()


if __name__ == "__main__":    main()