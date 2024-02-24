import cv2
import dlib
import glob
import numpy as np
import torch
import streamlit as st

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("scripts/shape_predictor_68_face_landmarks_GTX.dat")


def extract_lip_coordinates(detector, predictor, img_path):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (600, 500))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray)
    retries = 3
    while retries > 0:
        try:
            assert len(rects) == 1
            break
        except AssertionError as e:
            retries -= 1

    for rect in rects:
        # apply the shape predictor to the face ROI
        shape = predictor(gray, rect)
        x = []
        y = []
        for n in range(48, 68):
            x.append(shape.part(n).x)
            y.append(shape.part(n).y)
    return [x, y]


@st.cache_data(show_spinner=False, persist=True)
def generate_lip_coordinates(frame_images_directory):
    frames = glob.glob(frame_images_directory + "/*.jpg")
    frames.sort()

    img = cv2.imread(frames[0])
    height, width, layers = img.shape

    coords = []
    for frame in frames:
        x_coords, y_coords = extract_lip_coordinates(detector, predictor, frame)
        normalized_coords = []
        for x, y in zip(x_coords, y_coords):
            normalized_x = x / width
            normalized_y = y / height
            normalized_coords.append((normalized_x, normalized_y))
        coords.append(normalized_coords)
    coords_array = np.array(coords, dtype=np.float32)
    coords_array = torch.from_numpy(coords_array)
    return coords_array
