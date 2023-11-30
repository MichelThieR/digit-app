import base64
import json
import os
import re
import time
import uuid
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import cv2 as cv
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from svgpathtools import parse_path

import SimpleConvoModel
import torch
from torch import nn, optim
from torchvision import transforms

# linearNet = LinearpyTeen.PyTeen()
ConvoNet = SimpleConvoModel.PyTeen()

# function main contains the code for the main page
def main():
    if "button_id" not in st.session_state:
        st.session_state["button_id"] = ""
    if "color_to_label" not in st.session_state:
        st.session_state["color_to_label"] = {}

    full_app()

    with st.sidebar:
        st.markdown("---")
        st.markdown(
            '<h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp by <a href="https://twitter.com/andfanilo">@andfanilo</a></h6>',
            unsafe_allow_html=True,
        )


# full_app function corresponds to the canvas found in a public github repo
def full_app():
    st.sidebar.header("Configuration")
    st.markdown(
        """
    Draw a digit on the canvas, and the AI will guess it!
    * Go easy at first
    * Then try to fool it
    * We also appreciate feedback! :)
    """
    )

    
    # Specify canvas parameters in application
    drawing_mode = "freedraw" # limit user to one option
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
    # removing realtime update checkbox
    # realtime_update = st.sidebar.checkbox("Update in realtime", True)

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=18, # implicitly setting to 18
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=False, # update just sends the canvas image to streamlit
        height=400,
        width=400,
        drawing_mode=drawing_mode,
        point_display_radius=0,
        display_toolbar=st.sidebar.checkbox("Display toolbar", True),
        key="full_app",
    )

    # Do something interesting with the image data and paths
    if canvas_result.image_data is not None:
        # linearNet.load_state_dict(torch.load('LinearpyTeen30epocs.pth'))
        ConvoNet.load_state_dict(torch.load('pyTeenConvo9882.pth'))
        # st.image(canvas_result.image_data)
        cv.imwrite("canvas.png",  canvas_result.image_data)

        # this section of the code does the image manipulation
        # before passing it to our NN for guessing
        img = cv.imread("canvas.png",cv.IMREAD_GRAYSCALE)
        scaling = cv.resize(img,(28,28))
        my_img_processed = cv.bitwise_not(scaling) # inverting image (background needs to be black)
        cv.imwrite('new.png',my_img_processed)
        new_img = Image.open("new.png")
        # st.image(new_img)
        # my_img_transform = transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor()])
        # input_img = LinearpyTeen.My_transform(my_img_processed).unsqueeze(0) # converting it to a batch
        input_img = SimpleConvoModel.my_transform(my_img_processed).unsqueeze(0)
        # guessed_digit = linearNet.predict(input_img)
        guessed_digit = ConvoNet.predict(input_img)
        ans = str(guessed_digit.item())
        st.subheader(f"The AI guessed that it was a {ans}\n")

    # if canvas_result.json_data is not None:
    #     objects = pd.json_normalize(canvas_result.json_data["objects"])
    #     for col in objects.select_dtypes(include=["object"]).columns:
    #         objects[col] = objects[col].astype("str")
    #     st.dataframe(objects)


if __name__ == "__main__":

    st.set_page_config(
        page_title="Streamlit Guessing Digit Demo", page_icon=":pencil2:"
    )
    st.title("AI trying its best to guess a digit!")

    main()
