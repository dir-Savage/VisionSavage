import cv2
import numpy as np

class ImageProcessor:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Image at {image_path} could not be loaded.")

    def resize_image(self, width, height):
        return cv2.resize(self.image, (width, height))

    def convert_to_binary(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        return binary_image

    def show_bgr_channels(self):
        b, g, r = cv2.split(self.image)
        cv2.imshow("Blue Channel", b)
        cv2.imshow("Green Channel", g)
        cv2.imshow("Red Channel", r)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_saturation_channel(self):
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        cv2.imshow("Saturation Channel", s)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def apply_average_blur(self, kernel_size=(5, 5)):
        return cv2.blur(self.image, kernel_size)

    def apply_median_blur(self, kernel_size=5):
        return cv2.medianBlur(self.image, kernel_size)

    def apply_erosion(self, kernel_size=(5, 5), iterations=1):
        kernel = np.ones(kernel_size, np.uint8)
        return cv2.erode(self.image, kernel, iterations=iterations)

    def apply_dilation(self, kernel_size=(5, 5), iterations=1):
        kernel = np.ones(kernel_size, np.uint8)
        return cv2.dilate(self.image, kernel, iterations=iterations)

    def apply_morphological_operations(self, kernel_size=(5, 5), iterations=1):
        kernel = np.ones(kernel_size, np.uint8)
        dilated_image = cv2.dilate(self.image, kernel, iterations=iterations)
        final_image = cv2.erode(dilated_image, kernel, iterations=iterations)
        return final_image


def select_figure():
    print("Select a figure to process:")
    print("1. FigureOne.jpeg")
    print("2. FigureTwo.jpeg")
    print("3. FigureThree.jpeg")
    choice = input("Enter the number of the figure (1, 2, or 3): ")
    if choice == "1":
        return "figs/FigureOne.jpeg"
    elif choice == "2":
        return "figs/FigureTwo.jpeg"
    elif choice == "3":
        return "figs/FigureThree.jpeg"
    else:
        print("Invalid choice. Defaulting to FigureOne.jpeg.")
        return "figs/FigureOne.jpeg"


def select_filter(processor):
    print("\nSelect a filter to apply:")
    print("1. Resize Image")
    print("2. Convert to Binary")
    print("3. Show BGR Channels")
    print("4. Show Saturation Channel")
    print("5. Apply Average Blur")
    print("6. Apply Median Blur")
    print("7. Apply Erosion")
    print("8. Apply Morphological Operations (Dilation + Erosion)")
    choice = input("Enter the number of the filter (1-8): ")

    if choice == "1":
        width = int(input("Enter width: "))
        height = int(input("Enter height: "))
        resized_image = processor.resize_image(width, height)
        cv2.imshow("Resized Image", resized_image)
    elif choice == "2":
        binary_image = processor.convert_to_binary()
        cv2.imshow("Binary Image", binary_image)
    elif choice == "3":
        processor.show_bgr_channels()
    elif choice == "4":
        processor.show_saturation_channel()
    elif choice == "5":
        blurred_image = processor.apply_average_blur((5, 5))
        cv2.imshow("Blurred Image", blurred_image)
    elif choice == "6":
        median_blurred_image = processor.apply_median_blur(5)
        cv2.imshow("Median Blurred Image", median_blurred_image)
    elif choice == "7":
        eroded_image = processor.apply_erosion((5, 5), 1)
        cv2.imshow("Eroded Image", eroded_image)
    elif choice == "8":
        final_image = processor.apply_morphological_operations((5, 5), 1)
        cv2.imshow("Final Processed Image", final_image)
    else:
        print("Invalid choice. No filter applied.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Main Program
if __name__ == "__main__":
    try:
        # Select a figure
        figure_path = select_figure()

        # Initialize ImageProcessor with the selected figure
        processor = ImageProcessor(figure_path)

        # Select and apply a filter
        select_filter(processor)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cv2.destroyAllWindows()