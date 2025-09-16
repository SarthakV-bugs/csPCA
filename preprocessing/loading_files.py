import SimpleITK as sitk
import os


def check_image_dimensions(image_path):
    """
    This function checks and prints the dimensions of a 3D medical image.

    Args:
        image_path (str): The file path to the image.
    """
    # Check if the file exists before trying to read it
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
        return

    try:
        # Read the image using SimpleITK
        image = sitk.ReadImage(image_path)

        # Get the size of the image in pixels (x, y, z)
        dimensions = image.GetSize()

        # Print the dimensions in a clear format
        print(f"Image located at: {image_path}")
        print(f"Image dimensions (x, y, z): {dimensions}")

        # You can also get other information like the number of channels
        # For a grayscale image, GetNumberOfComponentsPerPixel() would be 1
        num_channels = image.GetNumberOfComponentsPerPixel()
        print(f"Number of channels: {num_channels}")

    except Exception as e:
        print(f"An error occurred while trying to read the image: {e}")


# --- Example Usage ---
# You need to replace this with the actual path to one of your image files.
# For example, something like:
# 'path/to/your/dataset/image_file.mha'
# 'data_dir/Case01/T2W.mha'
sample_image_path = '/home/ibab/PycharmProjects/csPCA/raw_csPCa_data/mri_images/fold0/picai_public_images_fold0/10029/10029_1000029_t2w.mha'

print("Checking image dimensions...")
check_image_dimensions(sample_image_path)