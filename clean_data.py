from PIL import Image, ImageOps


def is_mostly_black(image_path, threshold=10, black_percentage_threshold=0.9):
    """
    Delete the file if it contains mostly black pixels.
    """
    img = Image.open(image_path)
    img_gray = ImageOps.grayscale(img)
    pixels = img_gray.getdata()

    black_pixels_count = sum(1 for pixel_value in pixels if pixel_value <= threshold)
    total_pixels = len(pixels)

    if total_pixels == 0:
        return False

    black_percentage = black_pixels_count / total_pixels
    return black_percentage >= black_percentage_threshold


def move_files_to_data_dir():
    """
    Moves files to the data directory From the data temp directory.
    """
    import os
    import shutil

    source_dir = "./data_test"
    destination_dir = "./data"
    dead_letter_folder = "./to_delete"

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    if not os.path.exists(dead_letter_folder):
        os.makedirs(dead_letter_folder)

    for filename in os.listdir(source_dir):
        destination_file = os.path.join(destination_dir, filename)
        source_file = os.path.join(source_dir, filename)

        if os.path.isfile(source_file) and not is_mostly_black(source_file):
            # Move the file to the data directory
            shutil.move(source_file, destination_file)
        else:
            # Optionally, you can delete the file if it is mostly black
            shutil.move(source_file, dead_letter_folder)


if __name__ == "__main__":
    move_files_to_data_dir()
    # print(is_mostly_black("./data_test/w96_2.png"))
    # print(is_mostly_black("./data_test/3FO_BM02.png"))
