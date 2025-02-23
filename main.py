import cv2
import numpy as np
import sys
import time
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

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
        cv2.waitKey(10000)
        cv2.destroyAllWindows()

    def show_saturation_channel(self):
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        _, s, _ = cv2.split(hsv_image)
        cv2.imshow("Saturation Channel", s)
        cv2.waitKey(10000)  # Show for 15 seconds
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


def select_figure(console):
    console.print(Panel.fit("Select a figure to process (or press 0 to exit):", title="Figure Selection"))
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Option", style="cyan", justify="center")
    table.add_column("Figure", style="green")
    table.add_row("1", "FigureOne.jpeg")
    table.add_row("2", "FigureTwo.jpeg")
    table.add_row("3", "FigureThree.jpeg")
    table.add_row("0", "[bold red]Exit[/bold red]")  # Exit option
    console.print(table)

    choice = Prompt.ask("Enter the number of the figure", choices=["0", "1", "2", "3"], default="1")
    
    if choice == "0":
        console.print("[bold red]Exiting the program...[/bold red]")
        sys.exit(0)  # Exit the script
    
    return f"figs/FigureOne.jpeg" if choice == "1" else \
           f"figs/FigureTwo.jpeg" if choice == "2" else \
           f"figs/FigureThree.jpeg"


def select_filter(processor, console):
    console.print(Panel.fit("Select a filter to apply:", title="Filter Selection"))
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Option", style="cyan", justify="center")
    table.add_column("Filter", style="green")
    table.add_row("1", "Resize Image")
    table.add_row("2", "Convert to Binary")
    table.add_row("3", "Show BGR Channels (Combined)")
    table.add_row("4", "Show Saturation Channel")
    table.add_row("5", "Apply Average Blur")
    table.add_row("6", "Apply Median Blur")
    table.add_row("7", "Apply Erosion")
    table.add_row("8", "Apply Morphological Operations (Dilation + Erosion)")
    table.add_row("0", "[bold red]Exit[/bold red]")  # Exit option
    console.print(table)

    choice = Prompt.ask("Enter the number of the filter", choices=["0", "1", "2", "3", "4", "5", "6", "7", "8"], default="1")
    
    if choice == "0":
        console.print("[bold red]Exiting the program...[/bold red]")
        sys.exit(0)  # Exit the script

    processed_image = None

    if choice == "1":
        width = int(Prompt.ask("Enter width", default="120"))
        height = int(Prompt.ask("Enter height", default="300"))
        processed_image = processor.resize_image(width, height)
    elif choice == "2":
        processed_image = processor.convert_to_binary()
    elif choice == "3":
        b, g, r = cv2.split(processor.image)
        processed_image = cv2.merge([b, g, r])  # Merge channels into one image
    elif choice == "4":
        hsv_image = cv2.cvtColor(processor.image, cv2.COLOR_BGR2HSV)
        _, s, _ = cv2.split(hsv_image)
        processed_image = s
    elif choice == "5":
        processed_image = processor.apply_average_blur((5, 5))
    elif choice == "6":
        processed_image = processor.apply_median_blur(5)
    elif choice == "7":
        processed_image = processor.apply_erosion((5, 5), 1)
    elif choice == "8":
        processed_image = processor.apply_morphological_operations((5, 5), 1)

    if processed_image is not None:
        cv2.imshow("Processed Image", processed_image)
        cv2.waitKey(20000)  # Show for 20 seconds
        cv2.destroyAllWindows()


if __name__ == "__main__":
    console = Console()

    try:
        console.print(Panel.fit("Welcome to the Image Processor!", title="Image Processor", style="bold blue"))

        while True:  # Keep running until user exits
            figure_path = select_figure(console)
            processor = ImageProcessor(figure_path)
            select_filter(processor, console)

    except Exception as e:
        console.print(f"[bold red]An error occurred: {e}[/bold red]")
    finally:
        cv2.destroyAllWindows()
        sys.exit(0) 
