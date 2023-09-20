
#import necessary drives
from google.colab import drive
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt


                                ###PART 1###
# Step a: Mount the Google Drive
# This is responsible for mounting the google drive to the "Colab VM"
drive.mount('/content/drive', force_remount= True)

# Step b: Load the image into a numpy array
# To do this, we need to first provide a path to the image
image_path1 = '/content/drive/MyDrive/Images/eye1.jpg'
#Now we can read the image(s) into a numpy array
image1 = imread(image_path1)


############################################################
#PART 2 Images
image_path2 = '/content/drive/MyDrive/Images/balloon.png'
image_path3 = '/content/drive/MyDrive/Images/Dog_Breeds.jpg'
image_path4 = '/content/drive/MyDrive/Images/Ocean.jpg'

############################################################

# Step c: Print the sizes of the image (rows, columns, and planes)
# Explicitly unpack the shape of the image into rows, cols, and planes for better readability
rows, cols, planes = image1.shape
print(f"Image Sizes are: {rows} rows, {cols} columns, and {planes} color (RGB) planes")

# Step d: Create a grayscale image as the average of the RGB planes
# Calculate the mean along the 3rd axis which is the column planes
# aka knowing python is 0-indexed, we would poass in the axis as 2 which would
# access the "planes" index of the 2d array I mentioned before and average the
# RGB values for each pixel. We will also cast the float type integers (8bit)
# since that is what that values can range from (0 - 255)
grayscale_img = np.mean(image1, axis=2).astype(np.uint8)

# Step e: Find the brightest pixel on each row of the grayscale image
# Step f: Find the darkest pixel on each row of the grayscale image
# Compute column indices of brightest and darkest pixels for all rows in one turn
brightest_pixel = np.argmax(grayscale_img, axis = 1) #  finds the column index of the brightest pixel for each row.
darkest_pixel = np.argmin(grayscale_img, axis=1) # finds the column index of the darkest pixel for each row.

# Now we loop through each row to update the orginal image
for i in range(rows):
  # Color the brightest pixel red in the original image
  image1[i, brightest_pixel[i], :] = [255,0,0] #i is current row, brightest_pixel[i] is the col which has brightest pixel in that row, and : accesses all the color channels RGB
  # Color the darkest pixel blue in the original image
  image1[i, darkest_pixel[i], :] = [0, 0, 255]

# Step g: Display the resulting image
# Display the modified image
plt.imshow(image1)
plt.axis('off')  # turn off axes for better visualization
plt.show()


                                ###PART 2###

# Function to perform power law transformation on a single image for a single gamma value
def power_law_transform(image,gamma):

    # Convert the image to float for better precision
    image_float = image.astype('float')

    # Normalize the image to the range [0, 1]
    image_normalized = image_float / 255.0

    # Apply the power law transformation
    image_power_law = np.power(image_normalized, gamma)

    # Scale the image back to [0, 255] and convert to uint8
    image_output = (image_power_law * 255.0).astype('uint8')

    return image_output

image_paths = [image_path1, image_path2, image_path3, image_path4]

gamma_values = [0.5, 1, 2.2, 4.4]


# Now test and output original and power transfer'd one
# Loop through each image path
for image_path in image_paths:
    image = imread(image_path)

    # Initialize a plot to show the original and transformed images
    fig, axes = plt.subplots(1, len(gamma_values) + 1, figsize=(20, 10))

    # Plot the original image
    axes[0].imshow(image)
    axes[0].set_title(f"Original ({image_path.split('/')[-1]})")
    axes[0].axis('off')

    # Loop through each gamma value and apply the transformation
    for i, gamma in enumerate(gamma_values):
        transformed_image = power_law_transform(image, gamma)

        # Plot the transformed image
        axes[i+1].imshow(transformed_image)
        axes[i+1].set_title(f"Gamma = {gamma}")
        axes[i+1].axis('off')

    plt.show()








