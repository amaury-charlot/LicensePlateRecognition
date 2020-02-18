import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import correlate2d


lookup_table = {
	"0": "B", "1": "D", "2": "F", "3": "G", "4": "H",
	"5": "J", "6": "K", "7": "L", "8": "M", "9": "N",
	"10": "P", "11": "R", "12": "S", "13": "T", "14": "V",
	"15": "X", "16": "Z", "17": "0", "18": "1", "19": "2",
	"20": "3", "21": "4", "22": "5", "23": "6", "24": "7",
	"25": "8", "26": "9", "27": "-"
}


def find_vertical_bounds(hp, T):
	"""
	Finds the upper and lower bounds of the characters' zone on the plate based on threshold value T
	:param hp: horizontal projection (axis=1) of the plate image pixel intensities
	:param T: Threshold value for bound detection
	:return: upper and lower bounds
	"""

	N = len(hp)

	# Find lower bound
	i = 0
	while ~((hp[i] <= T) & (hp[i+1] > T)) & (i < int(N/2)):
		i += 1
	lower_bound = 0 if i == int(N/2) else i

	# Find superior bound
	i = N-1
	while ~((hp[i-1] > T) & (hp[i] <= T)) & (i > int(N/2)):
		i -= 1
	upper_bound = i

	return [lower_bound, upper_bound]


def find_horizontal_bounds(vp, T):
	"""
	Find bounds for each character on the plate for further segmentation of the characters based on threshold T.
	:param vp: Vertical projection (axis=0) of the plate image pixel intensities
	:return: List containing all characters' bounds
	"""
	N = len(vp)
	bool_bounds = (vp >= T)
	start_ind = 0
	end_ind = 1
	bounds = []
	for b in range(N-1):
		if bool_bounds[end_ind] & ~bool_bounds[start_ind]:  # search for upwards transition
			bounds.append(end_ind)
			last_bound = bounds[len(bounds) - 1]
			if end_ind - last_bound >= 99:
				bounds.append(last_bound + 98)
		start_ind += 1
		end_ind += 1

	# To ensure last character's width is inferior than test character's width
	if bounds:
		last_bound = bounds[len(bounds)-1]
		if end_ind - last_bound < 99:
			bounds.append(end_ind)
		else:
			bounds.append(last_bound + 98)

	# Return all characters' bounds
	return bounds


def find_all_indexes(input_str, search_str):
	"""
	Searches for substring/character in input_str
	:param input_str: String in which to search substring
	:param search_str: Substring to be searched
	:return: Indexes of all substring matching position
	"""
	l1 = []
	length = len(input_str)
	index = 0
	while index < length:
		i = input_str.find(search_str, index)
		if i == -1:
			return l1
		l1.append(i)
		index = i + 1
	return l1


def divide_characters(image, bounds):
	"""
	Extracts each characters, identify them and compose the full license plate number
	:param image: plate image
	:param bounds: bounds for each characters
	:return: license plate number
	"""
	N = len(bounds)
	plate_number = ""

	for i in range(N-1):
		character_image = image[:, bounds[i]:bounds[i+1]] # extract each character based on their bounds
		plate_number = plate_number + match_characters(character_image) # Compose full license plate
	
	# Check for issues with "-" character
	indexes = find_all_indexes(plate_number, "-") # Find indexes of all occurence of "-" in plate_number string
	N = len(indexes)
	M = len(plate_number)
	if N:
		if (N != 2) or (indexes[0] == 0) or (indexes[N-1] == M-1):
			return None

	# Check for plates with length != than 8 characters
	if M != 8:
		return None

	# Check for plates with consecutive "-" characters
	for i in range(N-1):
		if indexes[i] == indexes[i+1] - 1:
			return None

	# Return the final plate number
	return plate_number


def match_characters(character_image):
	"""
	Match each character extracted with the ones from the training set.
	Compute a score for each character tested and keep the one with the highest score.
	:param character_image: image of a character extracted from the plate
	:return: character matched
	"""

	character_image_width = character_image.shape[1]
	score = np.zeros(28)
	intermediate_score = []

	if character_image_width <= 98: # Check if character image is thinner than characters in the training set
		# Compute scores for Letters
		for i in range(17):
			file_path = "SameSizeLetters/" + str(i+1) + ".bmp"
			test_char = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) # Read training set characters
			test_character_width = test_char.shape[1]
			normalize_coef = test_char.shape[0] * character_image_width * 255 # Coef to normalize final score so that 0 < score < 1
			for start in range(min([test_character_width - character_image_width - 1, 2])): # Slide character image over test character
				crop_tc = test_char[:, start:start + character_image_width]
				#cv2.imshow('match', cv2.bitwise_not(cv2.bitwise_xor(crop_tc, character_image)))
				#cv2.waitKey(25)

				# Score is obtained by summing the result of bitwise NXOR operations between pixels in both images
				# NXOR returns a 1 when two pixels are the same and 0 when pixels are different
				# We then normalized the score by the score we would have obtained in the case of a perfect fit
				intermediate_score.append(np.sum(cv2.bitwise_not(cv2.bitwise_xor(crop_tc, character_image)))/normalize_coef)

			# Score for test character i is the max of scores for each sliding position over the test character
			score[i] = max(intermediate_score)
			intermediate_score.clear()

		# Compute scores for Numbers (same operation as for letters)
		for i in range(10):
			file_path = "SameSizeNumbers/" + str(i) + ".bmp"
			test_char = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
			test_character_width = test_char.shape[1]
			normalize_coef = test_char.shape[0] * character_image_width * 255
			for start in range(min([test_character_width - character_image_width - 1, 2])):
				crop_tc = test_char[:, start:start + character_image_width]
				#cv2.imshow('match', cv2.bitwise_not(cv2.bitwise_xor(crop_tc, character_image)))
				#cv2.waitKey(25)
				intermediate_score.append(np.sum(cv2.bitwise_not(cv2.bitwise_xor(crop_tc, character_image)))/normalize_coef)
			score[17 + i] = max(intermediate_score)
			intermediate_score.clear()

		# Test for bar character
		# Instead of sliding horizontally, the bar character is slid vertically
		# This is to prevent the case where bar characters are position at different heights
		test_char = create_bar_character(character_image.shape, 10, 15)  # bar character
		test_character_height = test_char.shape[0]
		normalize_coef = character_image.shape[0] * character_image_width * 255
		for start in range(test_character_height - character_image.shape[0] - 1):
			crop_tc = test_char[start:start + character_image.shape[0], :]
			#cv2.imshow('match', cv2.bitwise_not(cv2.bitwise_xor(crop_tc, character_image)))
			#cv2.waitKey(0)
			intermediate_score.append(np.sum(cv2.bitwise_not(cv2.bitwise_xor(crop_tc, character_image))) / normalize_coef)
		score[27] = max(intermediate_score)
		intermediate_score.clear()

		# Supplemental check to be sure character image is a real character
		sum_pix = np.sum(character_image)
		if sum_pix < 40000:
			return ""

		# print maximal score for each characters to compare and its index
		# print("Character matched is : ", lookup_table[str(np.argmax(score))])
		return lookup_table[str(np.argmax(score))]

	else:
		return ""


def create_bar_character(img_shape, bar_thickness, bar_width):
	"""
	Creates the image of a bar character
	:param img_shape: shape of the character to synthesize (height, width)
	:param bar_thickness: Thickness of the bar [pixels]
	:param bar_width: Width of the bar [pixels]
	:return: bar character image
	"""
	ch_height = img_shape[0] + 50
	ch_width = img_shape[1]
	bar = np.zeros((ch_height, ch_width), np.uint8)
	bart_init = int(ch_height / 2) - int(bar_thickness / 2)
	bart_end = bart_init + bar_thickness
	barw_init = 0
	barw_end = barw_init + bar_width
	bar[bart_init:bart_end, barw_init:barw_end] = 255 * np.ones([bar_thickness, min((bar_width, ch_width))])
	return bar


def compute_dft(img):
	"""
	Computes the dft of a gray scale image. Not used in the software.
	:param img: gray scale image
	:return: dft of image
	"""
	dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)  # compute dft of a gray_scale image
	dft_shift = np.fft.fftshift(dft)  # shift zero frequency from top left corner to center of the image
	magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

	plt.subplot(121), plt.imshow(img, cmap='gray')
	plt.title('Input Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
	plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
	plt.show()


def clean_borders(plate_image, epsilon):
	"""
	This function aims to clean the pixels close to the borders of the plate image.
	:param plate_image: plate image (gray scaled)
	:param epsilon: width of the cleaning zone around the borders (epsilon_h, epsilon_w)
	:return: cleaned plate image
	"""
	height = plate_image.shape[0]
	width = plate_image.shape[1]

	plate_image[0:epsilon[0], :] = 0
	plate_image[height - epsilon[0]:height, :] = 0
	plate_image[:, 0:epsilon[1]] = 0
	plate_image[:, width - epsilon[1]:width] = 0

	return plate_image


def segment_and_recognize(plate_img):
	"""
	Segment the plate and Recognize each character.
	:param plate_img: image of the plate to be analyzed
	:return: license plate number recognized
	"""
	# Clean image borders
	plate_img = clean_borders(plate_img, (4, 7)) # perfect was (4,7)

	# dilatation (to close gaps in the letters)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
	plate_img = cv2.dilate(plate_img, kernel, iterations=1)
	#plate_img = cv2.morphologyEx(plate_img, cv2.MORPH_CLOSE, kernel)

	# Find vertical bounds
	horizontal_project = np.sum(plate_img, axis=1)
	vertical_bounds = find_vertical_bounds(horizontal_project, 16800)

	# crop the upper and lower boundaries
	new_plate = plate_img[vertical_bounds[0]+1:vertical_bounds[1]][:]

	# reshape the plate image
	resize_factor = 85/new_plate.shape[0]
	dim = (int(new_plate.shape[1]*resize_factor), 85)
	new_plate = cv2.resize(new_plate, dim, interpolation=cv2.INTER_LINEAR)

	# Find characters horizontal bounds
	vertical_project = np.sum(new_plate, axis=0)
	horizontal_bounds = find_horizontal_bounds(vertical_project, 2000)
	if len(horizontal_bounds) < 6:  # 6 bounds = 5 characters, no plate has usually less than 5 characters
		return None

	# find vertical bounds
	img_width = new_plate.shape[1]
	img_height = new_plate.shape[0]

	# turn gray_scale to BGR
	new_plate = cv2.cvtColor(new_plate, cv2.COLOR_GRAY2BGR)

	# draw bounding lines
	for bnd in horizontal_bounds:
		new_plate = cv2.line(new_plate, (bnd, 0), (bnd, img_height), (0, 255, 0), 1)
	cv2.imshow('Plate image', new_plate)
	cv2.waitKey(25)

	# Display horizontal and vertical projections
	"""xx = np.arange(len(vertical_project))
	plt.figure(figsize=(20, 3))
	color = (214/255, 39/255, 40/255)
	plt.plot(xx, vertical_project, color=color)
	#ax = plt.gca()
	#ax.set_ylim(ax.get_ylim()[::-1])
	plt.show()"""

	# Change back from BGR to Gray scale
	new_plate = cv2.cvtColor(new_plate, cv2.COLOR_BGR2GRAY)

	# Divide and match all characters in the plate
	plate_number = divide_characters(new_plate, horizontal_bounds)
	# Return the plate number
	return plate_number
