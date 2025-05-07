import os

def list_directory_tree(startpath, indent=""):
    """
    Prints the tree structure of a directory.

    Args:
        startpath (str): The path of the directory to list.
        indent (str): The indentation string for the current level.
    """
    # Define markers for directories and files
    dir_marker = "├── "
    file_marker = "│   └── "
    last_item_marker = "└── "

    try:
        items = sorted(os.listdir(startpath)) # Get sorted list of items
    except PermissionError:
        print(f"{indent}{dir_marker}[Access Denied]: {os.path.basename(startpath)}")
        return
    except FileNotFoundError:
        print(f"Error: Directory not found at {startpath}")
        return

    for i, item_name in enumerate(items):
        path = os.path.join(startpath, item_name)
        is_last = (i == len(items) - 1)
        current_marker = last_item_marker if is_last else dir_marker

        if os.path.isdir(path):
            print(f"{indent}{current_marker}{item_name}/")
            # For the next level, extend the indent based on whether this was the last item
            next_indent = indent + ("    " if is_last else "│   ")
            list_directory_tree(path, next_indent)
        else:
            # Use a different marker for files, or simply print the name
            file_display_marker = last_item_marker if is_last else file_marker
            # For simplicity, we'll use the same marker logic as directories for files for now,
            # but adjust the prefix to indicate it's a file.
            # A more sophisticated tree would use different line connectors for files vs. last directories.
            print(f"{indent}{current_marker}{item_name}")


if __name__ == "__main__":
    # Get the present working directory
    current_directory = os.getcwd()
    print(f"Directory Tree for: {current_directory}")
    list_directory_tree(current_directory)
