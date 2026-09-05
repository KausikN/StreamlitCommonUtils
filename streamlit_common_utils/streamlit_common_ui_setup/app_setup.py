# Imports
import streamlit as st

from .app_setup_utils import *

# Main Functions
# UI App Setup

def build_sidebar_app(
        ui_config,
        page_funcs=None,
        common_args=None, page_funcs_args=None,
        show_home_page=True,
        sidebar_title="Choose one of the following",
        
    ):
    """
    Build a standard sidebar-based Streamlit app shell.
    """
    args = common_args if common_args is not None else {}
    page_funcs_args = page_funcs_args if page_funcs_args is not None else {}
    page_funcs = page_funcs or {}
    modes = ui_config.get("PROJECT_MODES", [])
    if show_home_page: modes = [ui_config.get("PROJECT_NAME")] + modes

    selected_box = st.sidebar.selectbox(sidebar_title, modes)

    if show_home_page and selected_box == ui_config.get("PROJECT_NAME"):
        build_home_page(ui_config)
    else:
        corresponding_func_name = selected_box.replace(" ", "_").lower()
        if corresponding_func_name in page_funcs:
            page_func_args = page_funcs_args.get(corresponding_func_name, {})
            page_funcs[corresponding_func_name](selected=selected_box, **args, **page_func_args)


def build_home_page(ui_config):
    """
    Render the default home page for a Streamlit app.
    """
    st.title(ui_config.get("PROJECT_NAME"))
    st.markdown("Github Repo: " + "[" + ui_config.get("PROJECT_LINK", "") + "](" + ui_config.get("PROJECT_LINK", "") + ")")
    st.markdown(ui_config.get("PROJECT_DESC", ""))
