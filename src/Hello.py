import os
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Stock Prophet! 👋")

st.sidebar.success("Here you can find various tools to help you with your stock analysis.")

st.markdown(
    """
    StockProphet is a web app that allows you to analyze stock data and predict future stock prices.
    
    **👈 Select a tab from the sidebar** to begin your stock analysis and find out what StockProphet can do.
    
    ### Want to learn more?
    """
)

logo_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'Logo.jpg')
if os.path.exists(logo_path):
    image = Image.open(logo_path)
    desired_width = 400
    aspect_ratio = image.size[1] / image.size[0]
    desired_height = int(desired_width * aspect_ratio)
    st.image(image, caption='Company Logo', width=desired_width, output_format='PNG')