import streamlit as st

video_file = open("../../RESOURCES\helmet.mp4", 'rb')
video_bytes = video_file.read()

st.video(video_bytes)