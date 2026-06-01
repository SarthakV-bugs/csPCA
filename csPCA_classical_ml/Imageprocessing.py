# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import SimpleITK as sitk
# from skimage.feature import graycomatrix, graycoprops
#
# # Load and preprocess the image
# image = sitk.ReadImage("/home/ibab/PycharmProjects/mlproject_data/10000/10000_1000000_adc.mha")
# image_array = sitk.GetArrayFromImage(image)
#
# # Select the middle slice
# middle_slice = image_array[image_array.shape[0] // 2]
#
# # Normalize pixel values to 0-255
# normalized = ((middle_slice - np.min(middle_slice)) / (np.max(middle_slice) - np.min(middle_slice)) * 255).astype(np.uint8)
#
# # Apply CLAHE
# # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# # enhanced = clahe.apply(normalized)
# #
# # # Resize properly
# # resized = cv2.resize(enhanced, (256, 256), interpolation=cv2.INTER_LINEAR)
# #
# # # Extract GLCM Texture Features
# # def extract_glcm_features(image):
# #     glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
# #     contrast = graycoprops(glcm, 'contrast')[0, 0]
# #     correlation = graycoprops(glcm, 'correlation')[0, 0]
# #     energy = graycoprops(glcm, 'energy')[0, 0]
# #     homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
# #     return [contrast, correlation, energy, homogeneity]
# #
# # # Extract features from the middle slice
# # features = extract_glcm_features(resized)
# # print("Extracted Features:", features)
# #
# # # Save the enhanced image
# # cv2.imwrite("enhanced_adc.png", resized)
# # print("Saved as enhanced_adc.png")
#
# import SimpleITK as sitk
# import numpy as np
# import cv2
# from skimage.feature import graycomatrix, graycoprops
#
# # Load the .mha file
# image = sitk.ReadImage("/home/ibab/PycharmProjects/mlproject_data/10000/10000_1000000_adc.mha")
#
# # Convert to NumPy array
# image_array = sitk.GetArrayFromImage(image)  # Shape: (31, 114, 116)
#
# # Normalize voxel intensities (0-255)
# image_array = ((image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array)) * 255).astype(np.uint8)
#
# # Compute Intensity Features
# def extract_intensity_features(image_array):
#     mean_intensity = np.mean(image_array)
#     variance = np.var(image_array)
#     skewness = np.mean((image_array - mean_intensity) ** 3) ** (1/3)
#     kurtosis = np.mean((image_array - mean_intensity) ** 4) ** (1/4)
#     return [mean_intensity, variance, skewness, kurtosis]
#
# # Compute Texture Features using 3D GLCM
# def extract_texture_features(image_array):
#     # Compute GLCM for the middle slice
#     middle_slice = image_array[image_array.shape[0] // 2]
#     glcm = graycomatrix(middle_slice, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
#     contrast = graycoprops(glcm, 'contrast')[0, 0]
#     energy = graycoprops(glcm, 'energy')[0, 0]
#     homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
#     correlation = graycoprops(glcm, 'correlation')[0, 0]
#     return [contrast, energy, homogeneity, correlation]
#
# # Compute Shape Features (e.g., Volume)
# def extract_shape_features(image_array):
#     threshold = np.percentile(image_array, 90)  # Keep top 10% brightest voxels
#     binary_mask = image_array > threshold
#     volume = np.sum(binary_mask)  # Count nonzero voxels
#     return [volume]
#
# # Extract all features
# intensity_features = extract_intensity_features(image_array)
# texture_features = extract_texture_features(image_array)
# shape_features = extract_shape_features(image_array)
#
# # Combine features
# all_features = intensity_features + texture_features + shape_features
#
# print("Extracted Features:", all_features)


import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

# Load the NIfTI image
image = sitk.ReadImage("/home/ibab/Downloads/ProstateX-0000_07-07-2011.nii.gz")
image_array = sitk.GetArrayFromImage(image)  # Convert to NumPy array
print(np.unique(image_array))  # Print unique pixel values

# Print shape (typically 3D)
print("Image Shape:", image_array.shape)  # (Slices, Height, Width)

# Show middle slice (for 3D images)
middle_slice = image_array[image_array.shape[0] // 2]

# Save image
plt.imshow(middle_slice, cmap="gray")
plt.axis("off")
plt.title("Middle Slice of the NIfTI Image")
plt.savefig("ProstateX_0000_middle_slice.png", bbox_inches="tight", dpi=300)  # Save as PNG
print("Saved as ProstateX_0000_middle_slice.png")

