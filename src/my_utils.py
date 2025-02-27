import os

def create_folder_if_not_exists(folder_path):
    """
    Creates a folder at the specified path if it does not already exist.

    Parameters:
    - folder_path (str): The path of the folder to create.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")


def set_mystyle(ax, size_text_ticks=8, alpha=.1):

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(.5)
    ax.spines['left'].set_linewidth(.5)
    ax.tick_params(axis="both", width=.5, labelsize=size_text_ticks)