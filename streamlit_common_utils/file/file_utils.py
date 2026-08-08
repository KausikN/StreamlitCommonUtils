# Imports
import os
import shutil

from .file_classes import *

# Main Functions
def check_data_same(src, dst):
    '''
    Check if data in src and dst are same.
    '''
    if (not os.path.exists(src)) or (not os.path.exists(dst)):
        return False

    if os.path.isfile(src) and os.path.isfile(dst):
        with open(src, "r", encoding="utf8") as src_file:
            src_content = src_file.read()
        with open(dst, "r", encoding="utf8") as dst_file:
            dst_content = dst_file.read()
        return src_content == dst_content

    if os.path.isdir(src) and os.path.isdir(dst):
        for root, dirs, files in os.walk(src):
            for file in files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dst, src_file.replace(src, "", 1))
                if (not os.path.exists(dst_file)):
                    return False
                if os.path.isfile(src_file) and os.path.isfile(dst_file):
                    with open(src_file, "r", encoding="utf8") as src_handle:
                        src_content = src_handle.read()
                    with open(dst_file, "r", encoding="utf8") as dst_handle:
                        dst_content = dst_handle.read()
                    if src_content != dst_content:
                        return False
        return True

    return False


def cascade_copy_path(path, save_parent, save_path, overwrite=True):
    '''
    Copy path to save_path (cascade copy).
    '''
    if not overwrite:
        if os.path.exists(os.path.join(save_parent, save_path)):
            return

    save_split = save_path.split("/")
    save_dir_path = "/".join(save_split[:-1]).rstrip("/")
    os.makedirs(os.path.join(save_parent, save_dir_path), exist_ok=True)

    if os.path.isfile(path):
        shutil.copy(path, os.path.join(save_parent, save_path))
    elif os.path.isdir(path):
        shutil.copytree(path, os.path.join(save_parent, save_path), dirs_exist_ok=True)


def cascade_remove_path(path, remove_parent, remove_path, check_edited=False):
    '''
    Remove path from remove_path (cascade remove).
    '''
    if check_edited:
        if not check_data_same(path, os.path.join(remove_parent, remove_path)):
            return

    full_remove_path = os.path.join(remove_parent, remove_path)
    remove_dirs = os.path.split(remove_path)[0]

    if os.path.exists(full_remove_path):
        if os.path.isfile(full_remove_path):
            os.remove(full_remove_path)
        elif os.path.isdir(full_remove_path):
            shutil.rmtree(full_remove_path)
        if remove_dirs.strip() != "":
            try:
                os.removedirs(os.path.join(remove_parent, remove_dirs))
            except Exception:
                pass
