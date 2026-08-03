import cv2

# Fonction qui sera appelée au clic de souris
def get_pixel_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # clic gauche
        print(f"Coordonnées en pixels : ({x}, {y})")

# Charger l'image
image_path = r"C:\Users\az03810\OneDrive - Alliance\Bureau\ia_detection\ia_detection\output\20230311_085028\event_20230311_085028_instant77.25_offset2.28448728997202\20230311_085028_485_tplus1.5s_straight_processed.jpg"
img = cv2.imread(image_path)

# Vérifier si l'image est chargée
if img is None:
    print("Erreur : impossible de charger l'image.")
    exit()

# Nom de la fenêtre
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", get_pixel_coordinates)

# Boucle d'affichage
while True:
    cv2.imshow("Image", img)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # appuyer sur Échap pour quitter
        break

cv2.destroyAllWindows()