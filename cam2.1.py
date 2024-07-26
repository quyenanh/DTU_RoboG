from ugot import ugot
import cv2
import numpy as np

# Initialize UGOT
got = ugot.UGOT()
got.initialize('10.220.5.228')
got.open_camera()
(w, h) = (640,240)  # Resolution
# Function to update Canny thresholds based on trackbar positions
def update_canny(val):
    global low_threshold, high_threshold
    low_threshold = cv2.getTrackbarPos('Low Threshold', 'frame')
    high_threshold = cv2.getTrackbarPos('High Threshold', 'frame')
def update_thre(val):
    global sthreshold
    sthreshold = cv2.getTrackbarPos('Threshold', 'frame')
# Create a window to display the frames
cv2.namedWindow('frame')

# Initialize Canny thresholds
low_threshold = 50
high_threshold = 150
sthreshold = 150

cv2.createTrackbar('Threshold', 'frame', sthreshold, 255, update_thre)

# Create trackbars for Canny edge detection thresholds
cv2.createTrackbar('Low Threshold', 'frame', low_threshold, 255, update_canny)
cv2.createTrackbar('High Threshold', 'frame', high_threshold, 255, update_canny)
try:
    while True:
        frame = got.read_camera_data()
        if frame is not None:
            nparr = np.frombuffer(frame, np.uint8)
            data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            # Convert to grayscale
            gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian Blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            cv2.imshow("data", data)
            # Apply binary thresholding
            _, thresholded = cv2.threshold(blurred, sthreshold, 255, cv2.THRESH_BINARY_INV)
            cv2.imshow("thresholded", thresholded)
            # Detect edges using Canny
            edges = cv2.Canny(thresholded, low_threshold, high_threshold)
            print(low_threshold,high_threshold)
            cv2.imshow("edges", edges)
            # Detect lines using Hough Transform
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

            if lines is not None:
                for line in lines:
                    for x1, y1, x2, y2 in line:
                        cv2.line(data, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Draw circles at the endpoints of the line
                        cv2.circle(data, (x1, y1), 5, (255, 0, 0), -1)
                        cv2.circle(data, (x2, y2), 5, (255, 0, 0), -1)

            cv2.imshow("frame", data)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    print('-----KeyboardInterrupt')
finally:
    cv2.destroyAllWindows()
