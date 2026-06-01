import SimpleITK as sitk
import numpy as np



def preprocessing():

    #load .mha image
    image = sitk.ReadImage("/home/ibab/PycharmProjects/mlproject_data/10000/10000_1000000_adc.mha")
    array = sitk.GetArrayFromImage(image)
    print(array.shape)  #shape of the image

    #check the spacing of the image
    spacing = image.GetSpacing() #x,y,z
    print(f"Voxel spacing:x = {spacing[0]:.2f}, y = {spacing[1]:.2f}, z = {spacing[2]:.2f}")

    #Img has anisotropic spacing
    #2D slicing of the image

    ##feature extraction using bag of features or complex graph network???

def main():
    preprocessing()

if __name__ == '__main__':
    main()