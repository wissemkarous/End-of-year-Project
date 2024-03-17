import streamlit as st
import os
from utils.demo import load_video, ctc_decode
from utils.two_stream_infer import load_model
import os
from scripts.extract_lip_coordinates import generate_lip_coordinates
import options as opt

st.set_page_config(layout="wide")

model = load_model()

st.title("Lipreading final year project Demo")

st.info(
    "The inference speed is very slow on Huggingface spaces due to it being processed entirely on CPU ",
    icon="ℹ️",
)
st.info("Author ©️ : wissem karous ")
st.info("Made with ❤️  ")
# Generating a list of options or videos
options = os.listdir(os.path.join("app_input"))
selected_video = st.selectbox("Choose video", options)

col1, col2 = st.columns(2)


with col1:
    file_path = os.path.join("app_input", selected_video)
    video_name = selected_video.split(".")[0]
    os.system(f"ffmpeg -i {file_path} -vcodec libx264 {video_name}.mp4 -y")

    # Rendering inside of the app
    video = open(f"{video_name}.mp4", "rb")
    video_bytes = video.read()
    st.video(video_bytes)


with col1, st.spinner("Splitting video into frames"):
    video, img_p, files = load_video(f"{video_name}.mp4", opt.device)
    prediction_video = video
    st.markdown(f"Frames Generated:\n{files}")
    frames_generated = True
with col1, st.spinner("Generating Lip Landmark Coordinates"):
    coordinates = generate_lip_coordinates(f"{video_name}_samples")
    prediction_coordinates = coordinates
    st.markdown(f"Coordinates Generated:\n{coordinates}")
    coordinates_generated = True

with col2:
    st.info("Ready to make prediction!")
    generate = st.button("Generate")
    if generate:
        with col2, st.spinner("Generating..."):
            y = model(
                prediction_video[None, ...].to(opt.device),
                prediction_coordinates[None, ...].to(opt.device),
            )
            txt = ctc_decode(y[0])
            st.text(txt[-1])
