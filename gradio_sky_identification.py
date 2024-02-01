#Hannah Nguyen
#CS5330 - Lab 1: Sky Pixel Identification

import gradio as gr
import cv2
import numpy as np

# Function to apply histogram equalization on the V channel of the HSV image
# to enhance contrast and improve feature detection.
def histogram_equalization(hsv_image):
    # Separate the HSV channels
    h, s, v = cv2.split(hsv_image)
    # Apply histogram equalization to the V channel
    v_equalized = cv2.equalizeHist(v)
    # Merge the channels back into an HSV image
    hsv_equalized = cv2.merge((h, s, v_equalized))
    return hsv_equalized

# Function to calculate the threshold values for sky detection
# based on the top region of the image, which is assumed to contain the sky.
def calculate_thresholds_from_top(hsv_image, top_fraction=0.25):
    # Define the top region of the image for sampling sky pixels
    top_region = hsv_image[:int(hsv_image.shape[0] * top_fraction), :, :]
    
    # Calculate the average hue, saturation, and value in the top region
    # to determine the characteristic colors of the sky in the image.
    avg_hue = np.mean(top_region[:, :, 0])
    avg_sat = np.mean(top_region[:, :, 1])
    avg_val = np.mean(top_region[:, :, 2])

    # Initialize the limits to cover a broad range of sky colors.
    # Adjust the ranges based on the averages calculated above.
    lower_limit = np.array([0, 0, min(180, avg_val - 30)])
    upper_limit = np.array([180, max(255, avg_sat + 30), 255])

    # Predefined hue ranges for typical sky colors.
    blue_hue_range = (100, 140)
    orange_hue_range = (10, 50)

    # Adjust the hue range based on whether the average hue corresponds to
    # typical blue skies or orange sunset/sunrise skies.
    if avg_hue > blue_hue_range[0] and avg_hue < blue_hue_range[1]:
        lower_limit[0] = blue_hue_range[0]
        upper_limit[0] = blue_hue_range[1]
    elif avg_hue > orange_hue_range[0] and avg_hue < orange_hue_range[1]:
        lower_limit[0] = orange_hue_range[0]
        upper_limit[0] = orange_hue_range[1]

    # Extend the saturation limits to include clouds, which have high brightness
    # and low saturation.
    lower_limit[1] = 0  # Include low saturation values for cloud detection.
    upper_limit[1] = 255  # High saturation for clear skies or sunsets

    # Extend the upper value limit to ensure we capture the brightness of the sky and clouds.
    upper_limit[2] = 255

    return lower_limit, upper_limit

# Function to validate the sky detection mask and adjust thresholds if necessary.
# This ensures that the amount of sky detected is within reasonable bounds.
def validate_thresholds(mask, hsv_image, initial_lower_limit, initial_upper_limit):
    height, width = mask.shape
    total_pixels = height * width
    sky_pixels = cv2.countNonZero(mask)

    # Calculate the percentage of the image identified as sky.
    sky_percentage = sky_pixels / total_pixels

    # Define target bounds for the percentage of sky in the image.
    target_min_sky_percentage = 0.2
    target_max_sky_percentage = 0.95

    # If the sky percentage is within the target bounds, return the current mask.
    if target_min_sky_percentage <= sky_percentage <= target_max_sky_percentage:
        return mask

    # If the sky percentage is outside the target bounds, adjust thresholds (1e-6 to handle edge case and avoid zero devision).
    adjustment_ratio = target_min_sky_percentage / (sky_percentage + 1e-6) if sky_percentage < target_min_sky_percentage else target_max_sky_percentage / (sky_percentage + 1e-6)

    # Adjust the hue and saturation ranges based on the deviation from the target sky percentage.
    hue_adjustment = (initial_upper_limit[0] - initial_lower_limit[0]) * (adjustment_ratio - 1)
    sat_adjustment = (initial_upper_limit[1] - initial_lower_limit[1]) * (adjustment_ratio - 1)

    # Apply the adjusted limits to create a new mask.
    new_lower_limit = np.array([
        max(0, initial_lower_limit[0] - hue_adjustment),
        max(0, initial_lower_limit[1] - sat_adjustment),
        initial_lower_limit[2]
    ])
    new_upper_limit = np.array([
        min(180, initial_upper_limit[0] + hue_adjustment),
        min(255, initial_upper_limit[1] + sat_adjustment),
        initial_upper_limit[2]
    ])
    # Ensure new limits are type uint8 for mask creation.
    new_lower_limit = np.array(new_lower_limit, dtype=np.uint8)
    new_upper_limit = np.array(new_upper_limit, dtype=np.uint8)
    adjusted_mask = cv2.inRange(hsv_image, new_lower_limit, new_upper_limit)

    return adjusted_mask

# Function to find the most probable horizon line using the Hough Line Transform.
# It filters out significantly vertical lines and assumes the horizon is the highest
def find_horizon_line(edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=80, maxLineGap=10)
    if lines is not None:
        lines = [l[0] for l in lines if abs(l[0][0] - l[0][2]) > abs(l[0][1] - l[0][3])]
        # Sort lines by their midpoint y-coordinate
        lines.sort(key=lambda x: (x[1]+x[3])/2)
        # The horizon line is the one with the smallest y-coordinate midpoint
        horizon_line = lines[0]
        return horizon_line
    return None

# Function to create a mask that excludes everything below the detected horizon line.
# This helps to differentiate between sky and non-sky regions, particularly in cases
# where reflections or similar colors appear below the horizon.
def mask_below_horizon(image, horizon_line):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.line(mask, (horizon_line[0], horizon_line[1]), (horizon_line[2], horizon_line[3]), 255, thickness=5)
    # Fill below the line to exclude anything below it
    cv2.floodFill(mask, None, seedPoint=(0, horizon_line[1] + 1), newVal=255)
    return mask

# The main function that processes the image for Gradio.
# It applies the steps for sky detection and outputs an image
# showing the original and sky-detected regions side by side.
def process_image_for_gradio(image):
    # Convert to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Calculate thresholds from the top part of the image
    lower_limit, upper_limit = calculate_thresholds_from_top(hsv_image, top_fraction=0.2)
    
    # Ensure the limits are uint8 before using them in cv2.inRange
    lower_limit = np.array(lower_limit, dtype=np.uint8)
    upper_limit = np.array(upper_limit, dtype=np.uint8)
    
    # Initial mask based on the initially calculated thresholds
    initial_mask = cv2.inRange(hsv_image, lower_limit, upper_limit)
    
    # Validate and adjust the mask if necessary
    mask = validate_thresholds(initial_mask, hsv_image, lower_limit, upper_limit)
    
    # Apply morphological operations to remove small noise
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Apply edge detection to help find the horizon line
    edges = cv2.Canny(mask, 50, 150)
    
    # Find the horizon line
    horizon_line = find_horizon_line(edges)
    
    # Create a mask to exclude everything below the horizon
    if horizon_line is not None:
        horizon_mask = mask_below_horizon(image, horizon_line)
        # Combine the sky mask with the horizon mask
        mask = cv2.bitwise_and(mask, horizon_mask)
    
    # Bitwise-AND mask and original image to extract sky
    sky = cv2.bitwise_and(image, image, mask=mask)
    
    # Stack the original image and the result side by side
    result = np.hstack((image, sky))
    
    return result


# Gradio function that wraps around process_image_for_gradio
def gradio_interface(image):
    output_image = process_image_for_gradio(image)
    return output_image

# Define the Gradio interface
iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.components.Image(type="numpy"),  # Updated input definition
    outputs=gr.components.Image(type="numpy"),  # Updated output definition
    title="Sky Pixel Identification",
    description="Upload an image to identify sky pixels. The result will show the original image and sky pixels side by side."
)

# Launch the interface
iface.launch(share=True) 