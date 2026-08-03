import cv2
import pandas as pd
import os
import re
from pathlib import Path
from asammdf import MDF
import gzip,shutil
from mat73 import loadmat 

def get_time_offset(mf4_filename, matlab_filename):

    m =MDF(mf4_filename)
    # 1. Récupérer le 2e nom de channel
    channel_name = list(m.channels_db.keys())[0]


    data = m.get(channel_name, group =1, index =0)
    context_ts = data.timestamps[0]


    dst = f"{matlab_filename[:-4]}_unzipped.mat"

    with open(matlab_filename, "rb") as f:
        magic = f.read(2)

    if magic == b'\x1f\x8b':  # gzip magic bytes
        with gzip.open(matlab_filename, "rb") as f_in, open(dst, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    else:
        dst = matlab_filename  # already uncompressed, use directly

    capsule = loadmat(dst)

    caps_ts = capsule['internal_signals']['FusionIn']['swcFusionIn__fusion_eth_in']['fus_cam_obj_in_1']['Time'][0]
    time_offset = caps_ts - context_ts
    #print(f"Time offset (seconds): {time_offset}")

    return time_offset

def extract_captures_from_excel(excel_file, video_folder, output_folder="output"):    
    os.makedirs(output_folder, exist_ok=True)
    df = pd.read_excel(excel_file)
    road_shape = None
    for index, row in df.iterrows():
        try:
            video_name = str(row['Nom_du_log']).strip()
            instant_raw = row['Debut_event_en_DareDeevil_approx']
            rayon_droite = float(row['Rayon_courbure_right_debut_event']) if row['Rayon_courbure_right_debut_event'] != "-" else None
            rayon_gauche = float(row['Rayon_courbure_left_debut_event']) if row['Rayon_courbure_left_debut_event'] != "-" else None
            road_shape = (row['Road_Shape']) if row['Road_Shape'] != "-" else None
            #distance_lat = float(row['V_m_TargetYdistMeas_apres']) if row['V_m_TargetYdistMeas_apres'] != "-" else None
            #distance_long = float(row['V_m_Distance_Meas_apres']) if row['V_m_Distance_Meas_apres'] != "-" else None
        except (ValueError, TypeError, KeyError) as e:
            print(f"Error cappturing photos row {index}: {e}")
            continue




        match = re.search(r'(\d{8})_(\d{6})', video_name)
        if match:
            date_part = match.group(1)
            time_part = match.group(2)
            new_time_int = int(time_part) + 1
            new_time_int2 = int(time_part) + 2
            new_time_part = f"{new_time_int:06d}"
            new_time_part2 = f"{new_time_int2:06d}"
            video_name = f"{date_part}_{time_part}"
            video_name1 = f"{date_part}_{new_time_part}"
            video_name2 = f"{date_part}_{new_time_part2}"

        mat_file = None
        mf4_file = None
        video_path = None
        for video in os.listdir(video_folder):
            #print(video_name, video_name2)
            if video_name in video and not '.mat' in video and not '.MF4' in video :
                video_path = os.path.join(video_folder, video)
            elif video_name1 in video and video.endswith(('.mp4', '.avi', '.mkv', '.AVI')):
                video_path = os.path.join(video_folder, video) 
            elif video_name2 in video and video.endswith(('.mp4', '.avi', '.mkv', '.AVI')):
                video_path = os.path.join(video_folder, video)
                
            if video_name in video and '.mat' in video and 'unzipped' not in video:
                mat_file = os.path.join(video_folder, video)
            elif video_name1 in video and '.mat' in video and 'unzipped' not in video:
                mat_file = os.path.join(video_folder, video)
            elif video_name2 in video and '.mat' in video and 'unzipped' not in video:
                mat_file = os.path.join(video_folder, video)

            if video_name in video and '.MF4' in video:
                mf4_file = os.path.join(video_folder, video)
            elif video_name1 in video and '.MF4' in video:
                mf4_file = os.path.join(video_folder, video)
            elif video_name2 in video and '.MF4' in video:
                mf4_file = os.path.join(video_folder, video)
        

        try:
            offset = (get_time_offset(mf4_file, mat_file)) if mat_file and mf4_file else 0

            instant = float(instant_raw) + offset
    
        except (ValueError, TypeError):
            #print(f"Instant invalide pour {video_name}: {instant_raw}")
            continue
        
        if video_path :
            video_orig_name = f"{match.group(1)}_{match.group(2)}" if match else None
            capsule_folder = os.path.join(output_folder, video_orig_name)
            event_folder = os.path.join(capsule_folder, f"event_{video_orig_name}_instant{instant_raw}_offset{offset}")
            os.makedirs(capsule_folder, exist_ok=True)
            os.makedirs(event_folder, exist_ok=True)

        time_offsets = [-1.5,-1.0, -0.5, 0,0.5, 1.0,1.5]

        if not video_path:
            #print(f"Aucune vidéo trouvée pour {video_name}")
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            #print(f"Impossible d'ouvrir la vidéo {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)

        for dt in time_offsets:

            t_capture = instant + dt
            if t_capture < 0:
                continue  # pas d'images avant le début

            frame_number = int(round(fps * t_capture))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()

            if ret:
                label = f"t{dt:+.1f}s".replace("+", "plus").replace("-", "minus")
                capture_name = f"{video_orig_name}_{label}_{road_shape}.jpg"

                output_path = os.path.join(event_folder, capture_name)
                if not os.path.exists(output_path): 
                    cv2.imwrite(output_path, frame)
        cap.release()



#excel_file = r"C:\Users\az03810\OneDrive - Alliance\Bureau\ia_detection\Analyse_ACC_Overselection_206p242_V5 (3).xlsx"
#video_folder = r"C:\Users\az03810\OneDrive - Alliance\Bureau\ia_detection\ia_detection\capsules"
#output_folder = Path("output")
#output_folder2 = Path(r"C:\Users\az03810\OneDrive - Alliance\Bureau\ia_detection\ia_detection\output")
#extract_captures_from_excel(excel_file, video_folder, output_folder)