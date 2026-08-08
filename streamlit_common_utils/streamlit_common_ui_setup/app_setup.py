# Imports
import streamlit as st

from .app_setup_utils import *

# Main Functions
# UI App Setup

def build_sidebar_app(ui_config, page_funcs=None):
    """
    Build a standard sidebar-based Streamlit app shell.
    """
    page_funcs = page_funcs or {}

    selected_box = st.sidebar.selectbox(
        "Choose one of the following",
        tuple([ui_config.get("PROJECT_NAME")] + ui_config.get("PROJECT_MODES", [])),
    )

    if selected_box == ui_config.get("PROJECT_NAME"):
        build_home_page(ui_config)
    else:
        corresponding_func_name = selected_box.replace(" ", "_").lower()
        if corresponding_func_name in page_funcs:
            page_funcs[corresponding_func_name]()


def build_home_page(ui_config):
    """
    Render the default home page for a Streamlit app.
    """
    st.title(ui_config.get("PROJECT_NAME"))
    st.markdown("Github Repo: " + "[" + ui_config.get("PROJECT_LINK", "") + "](" + ui_config.get("PROJECT_LINK", "") + ")")
    st.markdown(ui_config.get("PROJECT_DESC", ""))
