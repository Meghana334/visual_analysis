import cv2
import numpy as np
from collections import Counter


def rgb_to_hex(rgb):
    """Convert RGB tuple to hex color code"""
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def get_dominant_colors(image_path, num_colors=10):
    """
    Extract dominant colors from an image

    Args:
        image_path: Path to the image file
        num_colors: Number of dominant colors to extract

    Returns:
        List of tuples containing (hex_code, rgb_tuple, percentage)
    """
    # Read the image
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Could not read the image file")

    # Convert BGR to RGB (OpenCV uses BGR by default)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Reshape image to be a list of pixels
    pixels = img_rgb.reshape(-1, 3)

    # Use K-means clustering to find dominant colors
    pixels_float = np.float32(pixels)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    _, labels, palette = cv2.kmeans(pixels_float, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Count the occurrences of each cluster
    _, counts = np.unique(labels, return_counts=True)

    # Sort colors by frequency
    indices = np.argsort(counts)[::-1]

    # Calculate percentages and create result list
    total_pixels = len(pixels)
    colors = []

    for idx in indices:
        rgb = palette[idx]
        percentage = (counts[idx] / total_pixels) * 100
        hex_code = rgb_to_hex(rgb)
        colors.append({
            'hex': hex_code,
            'rgb': tuple(rgb.astype(int)),
            'percentage': round(percentage, 2)
        })

    return colors


def mouse_callback_color_picker(image_path):
    """
    Interactive color picker - click on image to get color at that point
    """
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def pick_color(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Get the color at clicked position
            bgr = img[y, x]
            rgb = img_rgb[y, x]
            hex_code = rgb_to_hex(rgb)

            print(f"\nColor at position ({x}, {y}):")
            print(f"RGB: {tuple(rgb)}")
            print(f"HEX: {hex_code}")

            # Create a small window showing the picked color
            color_display = np.zeros((100, 200, 3), dtype=np.uint8)
            color_display[:] = bgr
            cv2.putText(color_display, hex_code, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow('Picked Color', color_display)

    cv2.namedWindow('Image - Click to pick color')
    cv2.setMouseCallback('Image - Click to pick color', pick_color)

    cv2.imshow('Image - Click to pick color', img)
    print("\nClick on the image to pick colors. Press 'q' to quit.")

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def create_color_palette(colors, width=600, height=100):
    """
    Create a visual palette showing the extracted colors
    """
    palette = np.zeros((height, width, 3), dtype=np.uint8)

    start_x = 0
    for color in colors:
        # Calculate width for this color based on percentage
        color_width = int(width * (color['percentage'] / 100))

        # BGR for OpenCV
        bgr = color['rgb'][::-1]

        # Fill the section
        palette[:, start_x:start_x + color_width] = bgr

        start_x += color_width

    return palette


# Example usage
if __name__ == "__main__":
    # Example 1: Extract dominant colors
    print("=== Example 1: Extract Dominant Colors ===")
    print("Replace 'your_image.jpg' with your actual image path\n")

    # Uncomment and modify the path below to use with your image
    image_path = "/home/meghana/Downloads/bc_asset1.jpg"
    colors = get_dominant_colors(image_path, num_colors=8)

    print(f"Top {len(colors)} dominant colors:")
    for i, color in enumerate(colors, 1):
        print(f"{i}. HEX: {color['hex']} | RGB: {color['rgb']} | {color['percentage']}%")

    # Create and display color palette
    palette = create_color_palette(colors)
    cv2.imshow('Color Palette', palette)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # Example 2: Interactive color picker
    print("\n=== Example 2: Interactive Color Picker ===")
    print("Uncomment the lines below and provide your image path\n")

    # Uncomment to use interactive picker
    # mouse_callback_color_picker("your_image.jpg")

    print("\nTo use this script:")
    print("1. Uncomment the relevant section")
    print("2. Replace 'your_image.jpg' with your image file path")
    print("3. Run the script")