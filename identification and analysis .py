import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

def openfile():
    # Read image file, no chinese
    root = tk.Tk()
    file_path = filedialog.askopenfilename()
    image = cv2.imread(file_path)

    # Display image
    cv2.imshow('image', image)
    return image, file_path
image = openfile()
image = image[0]
file_path = image[1]

#click for gray value, thresholding
def get_grayvalue(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    print('press q to quit')
    def mouse_click(event, x, y, flags, para):
        if event == cv2.EVENT_LBUTTONDOWN:
            print('PIX:', x, y)
            print("BGR:", image[y, x])
            print("GRAY:", gray[y, x])
            print("HSV:", hsv[y, x])
    if __name__ == '__main__':
        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", mouse_click)
        while True:
            cv2.imshow('Image', image)
            if cv2.waitKey() == ord('q'):
                break
    cv2.destroyAllWindows()
    return gray, hsv
gray, hsv = get_grayvalue(image)

threshold = input('Enter threshold for this image: ')

# Apply Medianblur and thresholding to image
Grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
Grayimg = cv2.medianBlur(Grayimg, 5)
ret, thresh = cv2.threshold(Grayimg, threshold, 255,cv2.THRESH_BINARY)

# Show thresholded image
cv2.imshow('thresholded image', thresh)

# Find contours in thresholded image
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
filtered_contours = [c for c in contours if cv2.contourArea(c) >= 200]

# Create list of bounding rectangles and size data
bounding_rectangles = []
StructureHeight = []
StructureWidth = []
StructureArea = []

# Iterate over contours
for c in contours:
    # Filter out contours with area less than 80 定义最小面积
    if cv2.contourArea(c) < 100:
        continue

    # Get bounding rectangle for contour
    x, y, w, h = cv2.boundingRect(c)
    bounding_rectangles.append((x, y, w, h))

    # Draw bounding rectangle on image
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Add text above rectangle
    cv2.putText(image, 'structure', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    #get data
    area = cv2.contourArea(c)
    StructureHeight.append(h)
    StructureWidth.append(w)
    StructureArea.append(area)

# Show image with bounding rectangles
cv2.imshow('image with bounding rectangles', image)

# Create a figure with 3 subplots
fig, ax = plt.subplots(1, 3)

# Loop over the data sets
for i, data in enumerate([StructureArea, StructureWidth, StructureHeight]):
    # Convert the data to a NumPy array
    data = np.array(data)

    # Compute the mean, median, and standard deviation
    mean = np.mean(data)
    median = np.median(data)
    std = np.std(data)

    # Create a histogram of the data and add it to the subplot
    ax[i].hist(data, bins=20)
    ax[i].set_xlabel('value(pixel)')
    ax[i].set_ylabel('Count')

    # Add a title to the subplot with the statistics
    ax[i].set_title('i: Mean: {:.2f}, Median: {:.2f}, Std: {:.2f}'.format(mean, median, std))

#output data
print("Data: StructureArea, StructureWidth and StructureHeight")
for i in [StructureArea, StructureWidth, StructureHeight]:
    print(i)

data_path = input('Enter sample name: ')
data_path_txt = data_path + '.txt'
with open(data_path_txt, 'w') as f:
    f.write(data_path + '\n')
    f.write('Threshold: ' + str(threshold) + '\n')
    f.write('Data: StructureArea, StructureWidth and StructureHeight\n')
    for i in [StructureArea, StructureWidth, StructureHeight]:
        f.write(str(i))
        f.write('\n')

# Save processed image
pic_path = data_path + '.jpg'
cv2.imwrite(pic_path, image)

# Wait for user to press a key
cv2.waitKey(0)
