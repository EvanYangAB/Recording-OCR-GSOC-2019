import cv2
import pytesseract
from PIL import Image

def image_to_string(image_array):
	gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
	text = pytesseract.image_to_string(gray)
	return text


def file_to_string(file_path):
	image = cv2.imread(path)
	return image_to_string(image)