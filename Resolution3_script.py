# Librairies

# Importations
import numpy as np
import os
import sys
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from datetime import date, datetime
from skimage import *
from skimage.measure import profile_line
from skimage.metrics import structural_similarity as ssim
from skimage.feature import match_template
from skimage.measure import profile_line
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import interp1d
import xml.etree.ElementTree as ET
import tkinter as tk
from tkinter import filedialog
import webbrowser
import git
from datetime import datetime
import xml.etree.ElementTree as ET

# Variables globales
INITIAL_DIR = '../Images/'
PC = 'Windows'  
CRITERIA = "16" # pour images resolution 3 
BRIGHTNESS_FACTOR = 1.16
TEMPLATE1_PATH = "template5.jpg"
TEMPLATE2_PATH = "template5bis.jpg"
DEBUG_MODE = True
DISTANCE_BETWEEN_VER_LINES = 15
if DEBUG_MODE:
    print("Running in debug mode.")

def select_image_file(initial_dir):
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Setting the initial directory based on the operating system
    if os.name == 'posix':  # Linux and macOS
        initial_dir = os.path.join(initial_dir, "Résolution3")  # Adjust based on your directory structure
    else:  # Windows
        initial_dir = initial_dir

    path2data = filedialog.askdirectory(parent=root, initialdir=initial_dir, title='Please select a directory')
    print("Selected Directory is:", path2data)
    
    if not path2data:
        print('No directory selected, please make sure to double-click.')
        sys.exit(0)

    infodate = os.path.basename(path2data)
    print('Date from folder name:', infodate)

    try:
        date = datetime.strptime(infodate[:8], '%Y%m%d')
    except ValueError:
        print('The folder name does not contain a valid date, please check the format.')
        sys.exit(1)

    date_formatted = date.strftime('%d-%m-%Y')
    results_dir = os.path.join(path2data, 'Resultats_Analyses')

    # Create "Resultats_Analyses" directory if it does not exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print("Created folder:", results_dir)
    else:
        print("Folder already exists:", results_dir)

    # Define the base file name
    base_filename = '16-IQI_TrouFondPlat_'
    path_xml = os.path.join(path2data, f'{base_filename}{date_formatted}.tif.profile.xml')

    # Parse the XML file to find the acquisition time
    try:
        tree = ET.parse(path_xml)
        root_xml = tree.getroot()
        time_acquired = next(root_xml.iter("TimeAcquired")).text
        date_for_grafana = f"{time_acquired[:10]} {time_acquired[11:19]}"
        print("Time acquired for Grafana:", date_for_grafana)
    except ET.ParseError as e:
        print("Failed to parse the XML file:", e)
        sys.exit(1)
    except StopIteration:
        print("TimeAcquired element not found in the XML.")
        sys.exit(1)

    # Construct the full file path for the TIF image
    filepath = os.path.join(path2data, f'{base_filename}{date_formatted}.tif')
    return filepath, results_dir

# Fonctions 
def load_image(image_path):
    """
    Charge une image en niveau de gris
    :param image_path: chemin de l'image
    :return: image en niveau de gris
    """
    image = mpimg.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image

def display_image(image):
    """
    Affiche une image
    :param image: image à afficher
    """
    print(image.shape)
    print(image.dtype)
    plt.imshow(image, cmap='gray')
    plt.show()

def preprocess_image(image):
    # Convertir de 16 bits à 8 bits si nécessaire
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    image_initial = image
    # Ajuster la luminosité
    brightened_image = image * BRIGHTNESS_FACTOR
    # S'assurer que les valeurs sont dans l'intervalle [0, 255]
    brightened_image = np.clip(brightened_image, 0, 255)
    # Ajuster le contraste
    image_contraste = exposure.adjust_gamma(brightened_image, gamma=30)
    image_preprocessed = image_contraste
    return image_initial, image_preprocessed

def template_matching(template1_path, template2_path, image):
    """
    Trouver le meilleur match entre le template et l'image
    :param template1: template 1
    :param template2: template 2
    :param image: image
    :return: coordonnées du meilleur match
    """
    template1 = load_image(template1_path)
    template2 = load_image(template2_path)

    template1 =cv2.cvtColor(template1, cv2.COLOR_BGR2GRAY)
    template2 =cv2.cvtColor(template2, cv2.COLOR_BGR2GRAY)

    ################# Template 1 #####################
    liste_temp_match1 = []
    result = match_template(image, template1)

    # Trouver l'emplacement maximal de la correspondance
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]

    # Dessiner un rectangle autour de la zone correspondante
    h, w = template1.shape
    if DEBUG_MODE:
        fig, ax = plt.subplots(1, figsize=(5, 5))
        ax.imshow(image, cmap='gray')
        rect = plt.Rectangle((x, y), w, h, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.title('Detected Template')
        plt.show()

    # Croper l'image selon le template matching
    cropped_image1 = image[y:y+h, x:x+w]

    # Ajouter l'image croppée à la liste
    liste_temp_match1.append(cropped_image1)
    if DEBUG_MODE:
        # Afficher l'image croppée pour vérification
        plt.figure(figsize=(5, 5))
        plt.imshow(cropped_image1, cmap='gray')
        plt.title('Cropped Image 1')
        plt.show()

    ################## Template 2 ######################
    liste_temp_match2 = []
    # Appliquer le template matching
    result = match_template(cropped_image1, template2)

    # Trouver l'emplacement maximal de la correspondance
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]

    # Dessiner un rectangle autour de la zone correspondante
    h, w = template2.shape
    if DEBUG_MODE:
        fig, ax = plt.subplots(1, figsize=(5, 5))
        ax.imshow(cropped_image1, cmap='gray')
        rect = plt.Rectangle((x, y), w, h, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.title('Detected Template')
        plt.show()

    # Croper l'image selon le template matching
    cropped_image2 = cropped_image1[y:y+h, x:x+w]

    # Ajouter l'image croppée à la liste
    liste_temp_match2.append(cropped_image2)
    if DEBUG_MODE:
        # Afficher l'image croppée pour vérification
        plt.figure(figsize=(5, 5))
        plt.imshow(cropped_image2, cmap='gray')
        plt.title('Cropped Image 2')
        plt.show()
    
    return cropped_image2

def normalize_img(img):
    """
    Normaliser une image
    :param img: image à normaliser
    :return: image normalisée
    """
    min = np.min(img)
    max = np.max(img)

    normalized_image = (img - min) / (max - min) * 255
    normalized_image = np.uint8(normalized_image)
    # Supposons que  image soit stockée dans la variable "image"
    min_value = np.min(normalized_image)
    max_value = np.max(normalized_image)
    if DEBUG_MODE:
        print(min_value)
        print(max_value)
        plt.figure(figsize=(5, 5))
        plt.imshow(normalized_image, cmap='gray')
        plt.title('Normalized Image')
        plt.show()
    return normalized_image

def mean_shift_segmentation(image):
    if image is None:
        print("Error: Image not found.")
        return

    # Convert from BGR to RGB for displaying
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define the criteria for stopping the algorithm
    # (type, max number of iterations, accuracy)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500, 0.2)

    # Apply the Mean Shift algorithm to find the clusters
    segmented_image = cv2.pyrMeanShiftFiltering(original_image, sp=20, sr=40, maxLevel=1, termcrit=criteria)

    if DEBUG_MODE:
        # Displaying the original and segmented images
        plt.figure(figsize=(15, 10))
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(original_image)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('Segmented Image Using Mean Shift')
        plt.imshow(segmented_image)
        plt.axis('off')
        plt.show()

    return segmented_image

def circle_detection(segmented_img):
    gray_img = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
    if gray_img is None:
        raise Exception("Error loading image")

    blurred_gray_img = cv2.GaussianBlur(gray_img, (9, 9), 2)

    circles = cv2.HoughCircles(blurred_gray_img, cv2.HOUGH_GRADIENT, dp=2, minDist=20,
                               param1=70, param2=20, minRadius=1, maxRadius=20)

    output_gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(output_gray_img, (x, y), r, (0, 255, 0), 4)
    else:
        print("No circles detected.")
        return None, None, output_gray_img

    output_gray_img_rgb = cv2.cvtColor(output_gray_img, cv2.COLOR_BGR2RGB)

    if DEBUG_MODE:
        plt.imshow(output_gray_img_rgb)
        plt.title('Detected Circles')
        plt.axis('off')
        plt.show()

    circles = sorted(circles, key=lambda x: x[1])
    filtered_circles = circles[2:]
    circle_centers_filtered = [(x, y) for x, y, r in filtered_circles]

    return filtered_circles, circle_centers_filtered, output_gray_img_rgb

# Function to extract horizontal intensity profiles
def extract_horizontal_profile(image, y):
    """Extracts a horizontal intensity profile at the given y-coordinate."""
    profile = profile_line(image, (y, 0), (y, image.shape[1] - 1), linewidth=7)
    #print(profile)
    return profile

def trouver_pics_maximaux(profil):
    # Trouver les pics maximaux avec une hauteur minimale de 0
    peaks, _ = find_peaks(profil, height=15, distance=DISTANCE_BETWEEN_VER_LINES, prominence=15)

    # Trier les pics par valeur décroissante
    pics_valeurs = profil[peaks]
    pics_indices_tries = peaks[np.argsort(pics_valeurs)[::-1]]

    # Sélectionner les 10 premiers pics maximaux
    top_10_pics_indices = pics_indices_tries[:11]
    top_10_pics_valeurs = profil[top_10_pics_indices]

    return top_10_pics_indices, top_10_pics_valeurs

def lines_horizontal_detection(image, circles_centers):
    lines_horizontal = []
    # Calculate distances between circle centers to determine the step
    delta_y = []
    for i in range(len(circles_centers)):
        for j in range(i + 1, len(circles_centers)):
            x1, y1 = circles_centers[i]
            x2, y2 = circles_centers[j]
            distance_y = abs(y2 - y1)
            if 49 <= distance_y <= 60:
                delta_y.append(distance_y)

    if not delta_y:
        median_delta_y = 55  # Default value if no suitable distances found
    else:
        median_delta_y = np.median(delta_y)

    # Draw 10 horizontal dashed lines starting from the y-coordinate of the first circle
    if circles_centers:
        start_x, start_y = circles_centers[0]

        for i in range(10):
            y = int(start_y + i * median_delta_y)
            lines_horizontal.append(y)
            cv2.line(image, (0, int(y)), (image.shape[1], int(y)), (0, 0, 255), 2, lineType=cv2.LINE_AA)
        if DEBUG_MODE:
            # Displaying the image with line
            plt.imshow(image, cmap='gray')
            plt.axhline(y, color='r', linestyle='--')
            plt.title(f"Image with Line at y = {y}")
            plt.axis('off')
            plt.show()
    
    return image, lines_horizontal

def extract_vertical_profile(image, x):
    """Extracts a vertical intensity profile at the given x-coordinate."""
    profile = profile_line(image, (0, x), (image.shape[0] - 1, x), linewidth=5)
    return profile

    
def lines_vertical_detection(filtered_circles, img, image_to_draw):
    # Extract the center coordinates of the first circle
    center_x, center_y, radius = filtered_circles[0]
    print(f"Center X: {center_x}, Center Y: {center_y}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    center_y = int(center_y)
    line_profile = extract_horizontal_profile(img, center_y)

    # Detect maximum peaks using trouver_pics_maximaux
    pics_maximaux_indices, pics_maximaux_valeurs = trouver_pics_maximaux(line_profile)

    # Filter out indices that are less than the center x-coordinate
    new_pics_maximum_index = [pic for pic in pics_maximaux_indices if pic < center_x + 10]

    """
    # Print the indices and values of the maximum peaks
    print("Indices of pics_maximaux:", new_pics_maximum_index)
    print("Values of pics_maximaux:", [line_profile[index] for index in new_pics_maximum_index])
    """
    if DEBUG_MODE:
        # Plotting
        plt.plot(line_profile, label='Intensity Profile')
        plt.scatter(new_pics_maximum_index, [line_profile[index] for index in new_pics_maximum_index], 
                    color='red', marker='+', label='Pics Maximaux')
        plt.title('Line Profile with Maximum Peaks')
        plt.xlabel('Pixel')
        plt.ylabel('Intensity')
        #plt.legend()
        plt.show()

    for pic_index in new_pics_maximum_index:
        cv2.line(image_to_draw, (pic_index, 0), (pic_index, image_to_draw.shape[0]), (0, 255, 0), 2)
    if DEBUG_MODE:
        # Display the image with vertical lines
        plt.imshow(image_to_draw, cmap='gray')
        plt.title('Image with Vertical Lines')
        plt.axis('off')
        plt.show()

    return image_to_draw, new_pics_maximum_index

def generate_pdf(images, image, lines_horizontal, lines_vertical, results_dir):
    profiles_horizontal = [extract_horizontal_profile(image, y) for y in lines_horizontal]
    profiles_vertical = [extract_vertical_profile(image, x) for x in lines_vertical]
    titles = ['Image initiale', 'Template Matching', 'Mean shift', 'Circle Detection', 'Lines Detection']

    pdf_path = os.path.join(results_dir, 'output_Res3.pdf')
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(20, 80))  # Crée une figure unique pour le PDF
        gs = gridspec.GridSpec(2 + len(profiles_horizontal) + len(profiles_vertical), 1)  # Adjust for the new annotation section
        
        # Image grid
        upper_grid = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[0])
        for i in range(5):
            ax = fig.add_subplot(upper_grid[i])
            ax.imshow(images[i], cmap='gray')
            ax.set_title(titles[i])
            ax.axis('off')
        
        # Horizontal profiles
        for idx, profile in enumerate(profiles_horizontal):
            ax = fig.add_subplot(gs[1 + idx])
            ax.plot(profile)
            ax.set_title(f'Profile of Horizontal Line {idx + 1}')
        
        # Vertical profiles
        base_index = 1 + len(profiles_horizontal)
        for idx, profile in enumerate(profiles_vertical):
            ax = fig.add_subplot(gs[base_index + idx])
            ax.plot(profile, color='red')
            ax.set_title(f'Profile of Vertical Line {idx + 1}')

        # Annotation section
        ax_note = fig.add_subplot(gs[-1])  # Last section of GridSpec for notes
        ax_note.axis('off')
        today = date.today()
        gener = today.strftime('%d-%b-%Y')
        my_file = os.path.basename(__file__)
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        my_fontsize = 15
        ax_note.annotate(f'Generated on {gener}', (0, 0.8), fontsize=my_fontsize)
        ax_note.annotate(f'File: {my_file}', (0, 0.5), fontsize=my_fontsize)
        ax_note.annotate(f'Commit: {sha}', (0, 0.2), fontsize=my_fontsize)

        plt.tight_layout()  # Optimise l'agencement des subplots
        pdf.savefig(fig)  # Sauvegarde la figure actuelle dans le PDF
        plt.close()

        webbrowser.open_new(pdf_path)
        
# Fonction Main 
def main():
    file_path, results_dir = select_image_file(INITIAL_DIR)
    image = load_image(file_path)
    print('Preprocessing image...')
    image_initial, image_preprocessed = preprocess_image(image)
    cropped_img = template_matching(TEMPLATE1_PATH, TEMPLATE2_PATH, image_preprocessed)
    normalized_img = normalize_img(cropped_img)
    #normalized_gray = cv2.cvtColor(normalized_img, cv2.COLOR_BGR2GRAY)
    img_profile_lines = np.copy(normalized_img)
    print('Segmenting image by mean shift...')    
    segmented_image = mean_shift_segmentation(normalized_img)
    img_lines = np.copy(segmented_image)
    img_vertical = np.copy(segmented_image)
    print('Detecting circles...')
    circles, circle_centers, img_with_circles = circle_detection(segmented_image)
    print('lines detection...')
    image_to_draw, lines_horizontal = lines_horizontal_detection(img_lines, circle_centers)
    image_with_all_lines, lines_vertical = lines_vertical_detection(circles, img_vertical, image_to_draw)
    images = [image_initial, cropped_img, segmented_image, img_with_circles, image_with_all_lines]
    print('Generating PDF...')
    generate_pdf(images, img_profile_lines,lines_horizontal, lines_vertical, results_dir)

if __name__ == '__main__':
    main()