import cv2
import numpy as np

# Create a black image
black_image = np.zeros((500, 500, 3), dtype=np.uint8)

# Display the black image
cv2.imshow("Test Window", black_image)

print("Press any key to see its code. Press ESC to exit.")

while True:
    key1 = cv2.waitKey(0) & 0xFF  # First key press
    key2 = cv2.waitKey(0) & 0xFF  # Second key press (if available)

    print(f"Key 1: {key1}, Key 2: {key2}")

    if key1 == 27 or key2 == 27:  # ESC key
        break

cv2.destroyAllWindows()
