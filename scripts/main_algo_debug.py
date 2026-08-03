from video_capture import extract_captures_from_excel
from lane_and_Car_debug import get_info_from_excel, identify_car, add_to_final_list,overselection,write_event_status_to_excel,debug_plotting_mode,generate_report,get_ego_id_from_excel
import numpy as np
import lane_and_Car_debug
from pathlib import Path 
import cv2
import pandas as pd
from openpyxl import load_workbook
import matplotlib.pyplot as plt
from openpyxl.drawing.image import Image
import sys
from pathlib import Path
import argparse
import subprocess



parser = argparse.ArgumentParser()

# ✅ Arguments positionnels
parser.add_argument("excel_path", help="Chemin du fichier Excel")
parser.add_argument("video_folder", help="Dossier des vidéos")
parser.add_argument("output_folder", help="Dossier de sortie")
# ✅ Argument optionnel
parser.add_argument(
    "--categories",
    type=str,
    default="1,2,3,4",
    help="Catégories à analyser (ex: 1,2,4)"
)
parser.add_argument(
    "--create-pdf",
    action="store_true",
    help="Créer un rapport PDF à la fin de l'analyse"
)
parser.add_argument(
    "--configure-rois",
    action="store_true",
    help="Lancer le sélecteur de ROI avant l'analyse"
)
args = parser.parse_args()
excel_file = args.excel_path
video_dir = args.video_folder
video_dir_name = Path(video_dir).name
output_folder = Path(args.output_folder)/f"output_{video_dir_name}"
print(output_folder)
categories = [int(c) for c in args.categories.split(",")]
create_pdf = args.create_pdf
configure_ROIs = args.configure_rois
jsonfile_name = output_folder/f"selected_rois_{video_dir_name}.json"


summary_rows_CAT1 = []
summary_rows_CAT2 = []
summary_rows_CAT3 = []
summary_rows_CAT4 = []
summary_all = [] 


if not output_folder.exists():
    print("Creating output folder...")
    extract_captures_from_excel(excel_file, video_dir, output_folder)
    
else:
        print("Output folder already exists. Skipping capture extraction.")

if not Path(jsonfile_name).exists():
    ret = subprocess.call([sys.executable, "roi_selector.py", excel_file,video_dir_name,str(output_folder)])
    if ret != 0:
        print("Error occurred while running roi_selector.py. Exiting.")
        sys.exit(1)
else :
    if configure_ROIs :
        ret = subprocess.call([sys.executable, "roi_selector.py", excel_file,video_dir_name,str(output_folder)])
        if ret != 0:
            print("Error occurred while running roi_selector.py. Exiting.")
            sys.exit(1)

print("Starting analysis of captured images...")
for folder in output_folder.glob("*"):
    
    for subfolder in folder.glob("*"):
        #print(f"subfolder: {subfolder}")
        fit_liste = []
        candidates_list = {}
        final_matches = {}
        best_match = None
        best_err1 = None
        shape_list =['straight','slight_curve_right','slight_curve_left','right_curve','left_curve']
        road_shape = 'straight'
        i=0
        ego_id = get_ego_id_from_excel(excel_file,subfolder)
        lane_quality_scores = []  # Accumuler les scores de qualité des voies
        

        for img_path in subfolder.glob("*.jpg"):
            if not img_path.name.endswith("_processed.jpg"):
                img = cv2.imread(str(img_path))
                
                if img is not None:
                    #print(f"Processing image: {img_path}")


############################################## LANE DETECTION #############################################                   
                    for shape in shape_list:
                        if shape in img_path.name.lower() :
                            road_shape = shape
                            break
                    if  road_shape != 'right_curve' and road_shape != 'left_curve':
                        original_frame = img
                        lane_obj = lane_and_Car_debug.Lane(orig_frame=original_frame,img_path=img_path,road_shape=road_shape,ego_id=ego_id,video_folder=video_dir_name, output_folder=output_folder)
                        lane_line_markings, avg_luminosity = lane_obj.white_tape_mask(plot=False)
                        lane_obj.plot_roi(plot=False)                  
                        warped_frame = lane_obj.perspective_transform(plot=False)
                        histogram = lane_obj.calculate_histogram(plot=False)  
                        left_fit, right_fit = lane_obj.get_lane_line_indices_sliding_windows(plot=False)
                        
                        if lane_obj.detected_lanes :
                            curr_left_fit, curr_right_fit,curr_left_fitx, curr_right_fitx, curr_ploty = lane_obj.get_lane_line_previous_window(left_fit, right_fit, plot=False)
                            lane_obj.calculate_curvature(print_to_terminal=False)
                            curr_left_fitx, curr_right_fitx= lane_obj.validate_lanes(curr_left_fitx, curr_right_fitx,road_shape=road_shape)
                            frame_with_lane_lines = lane_obj.overlay_lane_lines(plot=False)                          
                            lane_obj.calculate_car_position(print_to_terminal=False)
                            frame_with_lane_lines2 = lane_obj.display_curvature_offset(frame=frame_with_lane_lines, plot=False)
                            cv2.waitKey(0) 
                            cv2.destroyAllWindows() 
                            
                            if i >0 :
                                left_fit, right_fit, left_fitx,right_fitx, ploty = fit_liste[i-1]
                                
                                if abs(curr_left_fit[0]/left_fit[0]) > 10**2 :
                                    print('current left fit:',curr_left_fit,'prev left fit:',left_fit)
                                    print("LEFT Polynomial coefficients have changed significantly")
                                    lane_obj.left_fit, lane_obj.right_fit, lane_obj.left_fitx,lane_obj.right_fitx, lane_obj.ploty = fit_liste[i-1]
                                    frame_with_lane_lines = lane_obj.overlay_lane_lines(plot=False)
                                    lane_obj.calculate_curvature(print_to_terminal=False)                            
                                    lane_obj.calculate_car_position(print_to_terminal=False)
                                    frame_with_lane_lines2 = lane_obj.display_curvature_offset(frame=frame_with_lane_lines, plot=False)


                                elif abs(curr_right_fit[0]/right_fit[0]) > 10**2 :
                                    print('current right fit:',curr_right_fit,'prev left fit:',right_fit)
                                    print("Right Polynomial coefficients have changed significantly")
                                    lane_obj.left_fit, lane_obj.right_fit, lane_obj.left_fitx,lane_obj.right_fitx, lane_obj.ploty = fit_liste[i-1]
                                    frame_with_lane_lines = lane_obj.overlay_lane_lines(plot=False)
                                    lane_obj.calculate_curvature(print_to_terminal=False)                            
                                    lane_obj.calculate_car_position(print_to_terminal=False)
                                    frame_with_lane_lines2 = lane_obj.display_curvature_offset(frame=frame_with_lane_lines, plot=False)


                                elif abs(curr_right_fit[0]/right_fit[0]) > 10**2  and abs(curr_left_fit[0]/left_fit[0]) > 10**2:
                                    print("Polynomial coefficients have changed significantly. Correcting fit.")
                                    #lane_obj.left_fit, lane_obj.right_fit, lane_obj.left_fitx,lane_obj.right_fitx, lane_obj.ploty = fit_liste[i-1]
                                    lane_obj.roi_points = np.float32([
                                                        (570,423), # Top-left corner
                                                        (300, 603), # Bottom-left corner            
                                                        (1100,580), # Bottom-right corner
                                                        (700,414) # Top-right corner
                                                        ])
                                    lane_obj.white_tape_mask()
                                    lane_obj.plot_roi(plot=False)
                                    warped_frame = lane_obj.perspective_transform(plot=False)
                                    histogram = lane_obj.calculate_histogram(plot=False)  
                                    left_fit, right_fit = lane_obj.get_lane_line_indices_sliding_windows(plot=False)
                                    curr_left_fit, curr_right_fit,curr_left_fitx, curr_right_fitx, curr_ploty = lane_obj.get_lane_line_previous_window(left_fit, right_fit, plot=False)
                                    frame_with_lane_lines = lane_obj.overlay_lane_lines(plot=False)
                                    lane_obj.calculate_curvature(print_to_terminal=False)                            
                                    lane_obj.calculate_car_position(print_to_terminal=False)
                                    frame_with_lane_lines2 = lane_obj.display_curvature_offset(frame=frame_with_lane_lines, plot=False)
                            
                            fit_liste.append((curr_left_fit, curr_right_fit,curr_left_fitx, curr_right_fitx, curr_ploty))
                            size = len(str(img_path))
                            new_filename = str(img_path)[:size - 4]
                            new_filename = new_filename + '_processed.jpg'    
                            cv2.imwrite(new_filename, frame_with_lane_lines2)
                            lane_quality = lane_obj.lane_detection_quality()
                            lane_quality_scores.append(lane_quality)
                            #print(f"📊 Qualité détection voies: {lane_quality}/100")
                            cv2.waitKey(0) 
                            cv2.destroyAllWindows() 
                            i+=1
                        else :
                            print( 'No lanes detected')
                            lane_obj.detected_lanes = False
                            for j in range(2) :
                                if avg_luminosity <= 100 :
                                    print("Low luminosity detected. Adjusting threshold.")
                                    lane_obj.thresh_small -= 30
                            
                                else :
                                    print("High luminosity detected. Adjusting threshold.")
                                    lane_obj.thresh_big -= 30
                                print(f"New threshold: {'small',lane_obj.thresh_small,'big',lane_obj.thresh_big}")
                                lane_line_markings, avg_luminosity = lane_obj.white_tape_mask(plot=False)
                                lane_obj.plot_roi(plot=False)                  
                                warped_frame = lane_obj.perspective_transform(plot=False)
                                histogram = lane_obj.calculate_histogram(plot=False)  
                                left_fit, right_fit = lane_obj.get_lane_line_indices_sliding_windows(plot=False)
                                if left_fit is not None and right_fit is not None:
                                    curr_left_fit, curr_right_fit,curr_left_fitx, curr_right_fitx, curr_ploty = lane_obj.get_lane_line_previous_window(left_fit, right_fit, plot=False)
                                    lane_obj.calculate_curvature(print_to_terminal=False)
                                    curr_left_fitx, curr_right_fitx= lane_obj.validate_lanes(curr_left_fitx, curr_right_fitx,road_shape=road_shape)                    
                                    frame_with_lane_lines = lane_obj.overlay_lane_lines(plot=False)                           
                                    lane_obj.calculate_car_position(print_to_terminal=False)
                                    frame_with_lane_lines2 = lane_obj.display_curvature_offset(frame=frame_with_lane_lines, plot=False)
                                    cv2.waitKey(0) 
                                    cv2.destroyAllWindows() 
                                    fit_liste.append((curr_left_fit, curr_right_fit,curr_left_fitx, curr_right_fitx, curr_ploty))
                                    size = len(str(img_path))
                                    new_filename = str(img_path)[:size - 4]
                                    new_filename = new_filename + '_processed.jpg'    
                                    cv2.imwrite(new_filename, frame_with_lane_lines2)
                                    lane_quality = lane_obj.lane_detection_quality()
                                    lane_quality_scores.append(lane_quality)
                                    print(f"📊 Qualité détection voies: {lane_quality}/100")
                                    cv2.waitKey(0) 
                                    cv2.destroyAllWindows() 
                                    i+=1
                                    break

        
                                                        ##########################

                            if left_fit is None and right_fit is None:
                                print("creating fictive lane lines")
                                curr_left_fitx, curr_right_fitx, curr_left_fit, curr_right_fit, curr_ploty = lane_obj.create_fictive_lanes()
                                frame_with_lane_lines = lane_obj.overlay_lane_lines(plot=False)                           
                                lane_obj.calculate_car_position(print_to_terminal=False)
                                frame_with_lane_lines2 = lane_obj.display_curvature_offset(frame=frame_with_lane_lines, plot=False)
                                cv2.waitKey(0) 
                                cv2.destroyAllWindows() 
                                fit_liste.append((curr_left_fit, curr_right_fit,curr_left_fitx, curr_right_fitx, curr_ploty))
                                size = len(str(img_path))
                                new_filename = str(img_path)[:size - 4]
                                new_filename = new_filename + '_processed.jpg'    
                                cv2.imwrite(new_filename, frame_with_lane_lines2)
                                lane_quality = lane_obj.lane_detection_quality()
                                lane_quality_scores.append(lane_quality)
                                #print(f"⚠️  Voies fictives créées - Qualité: {lane_quality}/100")
                                cv2.waitKey(0) 
                                cv2.destroyAllWindows() 
                                i+=1
                                break

    ############################################ VEHICULE DETECTION + DISTANCE CALCULATION ##########################
                        candidates_list = lane_obj.analyze_output_images2(image_path=new_filename,plot=True,candidates_list=candidates_list)
                        


################################################ matching photos with excel data and calculating distance longitudinal and lateral  + OVERSELECTION LOGIC ##########################                    
        if  road_shape != 'right_curve' and road_shape != 'left_curve':
            longi_excel,lat_excel,event_truth_excel,reason_excel,category = get_info_from_excel(excel_file,subfolder=subfolder.name)
            for photo in candidates_list :
                if 'tplus0.0' in photo:
                    best_match,best_err1 = identify_car(longi_excel,lat_excel,candidates_list[photo])
                    #print (f"Best match for tplus0.0: {best_match}")
                    if best_match :
                        add_to_final_list(final_matches, 'tplus0.0', best_match)
                    break
            
            if not best_match :
                INIT_SEARCH = [
                    "tplus0.5", "tminus0.5",
                    "tplus1.0", "tminus1.0",
                    "tplus1.5", "tminus1.5"
                ]
                
                for photo in candidates_list :
                    #for key in INIT_SEARCH:
                    match, err = identify_car(longi_excel, lat_excel, candidates_list[photo])

                    if match is not None:
                        best_match = match
                        best_err1 = err  # ✓ Now properly tracking the valid match error
                        break
    
            T = [0.5, 1.0, 1.5]
            current_match_plus = best_match   # on commence avec t=0.0
            current_match_moins = best_match
            for t in T:
                t_minus_name = f"tminus{t:.1f}s"
                t_plus_name  = f"tplus{t:.1f}s"

                # Chercher du côté t- (si présent)
                for photo in candidates_list :

                    if t_minus_name in photo:
                        if current_match_moins :
                            match_minus, err_minus = identify_car(
                                current_match_moins['distance_longitudinale'], 
                                current_match_moins['distance_laterale'], 
                                candidates_list[photo]
                            )
                            #print(f"Best match at {t_minus_name}: {match_minus}")
                            if match_minus is not None:  # Only if valid match (error <= 17)
                                current_match_moins = match_minus    # mise à jour du match
                                if err_minus < best_err1:
                                    best_err1 = err_minus  # Track lowest error
                                add_to_final_list(final_matches, t_minus_name, match_minus)
                    else :
                        pass
                
                # Chercher du côté t+ (si présent)
                for photo in candidates_list :

                    if t_plus_name in photo:
                        if current_match_plus :
                            match_plus, err_plus = identify_car(
                                current_match_plus['distance_longitudinale'], 
                                current_match_plus['distance_laterale'], 
                                candidates_list[photo]
                            )
                            #print(f"Best match at {t_plus_name}: {match_plus}")
                            if match_plus is not None:  # Only if valid match (error <= 17)
                                current_match_plus = match_plus    # mise à jour du match
                                if err_plus < best_err1:
                                    best_err1 = err_plus  # Track lowest error
                
                                add_to_final_list(final_matches, t_plus_name, match_plus)
                        else:
                            pass
                
            with open(subfolder / "final.txt", "w") as f:
                for key, value in final_matches.items():
                    f.write(f"{key}: {value}\n")
                # Calculer la qualité moyenne de détection des voies
                avg_lane_quality = int(np.mean(lane_quality_scores)) if lane_quality_scores else 0
                #print(f"📊 Qualité moyenne détection voies: {avg_lane_quality}/100")
                reason_script, trust_score,event_truth_script = overselection(final_matches, f, road_shape)#, lane_quality_score=avg_lane_quality)
            summary_all.append([subfolder.name, road_shape,reason_script,reason_excel,event_truth_script,event_truth_excel,trust_score,best_err1,category])
            if category == 1:
                summary_rows_CAT1.append([subfolder.name, road_shape,reason_script,reason_excel,event_truth_script,event_truth_excel,trust_score,best_err1,category])
            elif category == 2:
                summary_rows_CAT2.append([subfolder.name, road_shape,reason_script,reason_excel,event_truth_script,event_truth_excel,trust_score,best_err1,category])
            elif category == 3:
                summary_rows_CAT3.append([subfolder.name, road_shape,reason_script,reason_excel,event_truth_script,event_truth_excel,trust_score,best_err1,category])
            elif category == 4:
                summary_rows_CAT4.append([subfolder.name, road_shape,reason_script,reason_excel,event_truth_script,event_truth_excel,trust_score,best_err1,category])

if 1 in categories:
    summary_file = debug_plotting_mode(summary_rows_CAT1, summary_all, output_folder,category=1)
if 2 in categories:
    summary_file = debug_plotting_mode(summary_rows_CAT2, summary_all, output_folder,category=2)
if 3 in categories:
    summary_file = debug_plotting_mode(summary_rows_CAT3, summary_all, output_folder,category=3)
if 4 in categories:
    summary_file = debug_plotting_mode(summary_rows_CAT4, summary_all, output_folder,category=4)

print("Analysis complete. saving results to Excel.")
write_event_status_to_excel(excel_file, summary_file)
print("Results saved in Excel ending with 'yourexcelfile_Updated.xlsx'")

if create_pdf:
    print("Creating PDF report...")
    image = 'renault_group.png'
    generate_report(summary_file, output_folder, image,output_folder.name)
    print("PDF report generated successfully.")
