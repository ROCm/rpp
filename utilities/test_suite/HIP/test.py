import numpy as np
import cv2
# red_channel = np.random.randint(7, 57, (150, 150), dtype=np.uint8)
# green_channel = np.random.randint(0, 114, (150, 150), dtype=np.uint8)
# blue_channel = np.random.randint(14, 171, (150, 150), dtype=np.uint8)

# # Combine color channels into an RGB image
# rgb = np.stack((red_channel, green_channel, blue_channel), axis=2)
# print("Min R, G, B: ", np.min(red_channel), np.min(green_channel), np.min(blue_channel))
# print("Max R, G, B: ", np.max(red_channel), np.max(green_channel), np.max(blue_channel))

image = cv2.imread("/media/utilities/test_suite/TEST_IMAGES/min_max_images/image2.jpg", 0)
image_numpy = np.asarray(image)
# blue_channel = image[:, :, 0]  # Blue channel
# green_channel = image[:, :, 1]  # Green channel
# red_channel = image[:, :, 2]  # Red channel

# Split the image into color channels
# blue_channel = image[:, :, 0]  # Blue channel
# green_channel = image[:, :, 1]  # Green channel
# red_channel = image[:, :, 2]  # Red channel

# Find the minimum value for each channel
# min_blue = cv2.minMaxLoc(blue_channel)[1]
# min_green = cv2.minMaxLoc(green_channel)[1]
# min_red = cv2.minMaxLoc(red_channel)[1]

print("Minimum value for Blue channel:", np.max(image_numpy))
# print("Minimum value for Green channel:", min_green)
# print("Minimum value for Red channel:", min_red)