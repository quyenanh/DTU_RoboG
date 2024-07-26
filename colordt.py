import cv2
from PIL import Image

from util import get_limits
from ugot import ugot
import numpy as np

# Initialize UGOT
got = ugot.UGOT()
got.initialize('10.220.5.228')
got.open_camera()

try:
    yellow = [0, 0, 0]  # yellow in BGR colorspace
    while True:
        frame = got.read_camera_data()
        if frame is not None:
            nparr = np.frombuffer(frame, np.uint8)
            data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            hsvImage = cv2.cvtColor(data, cv2.COLOR_BGR2HSV)

            lowerLimit, upperLimit = get_limits(color=yellow)

            mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

            mask_ = Image.fromarray(mask)

            bbox = mask_.getbbox()

            if bbox is not None:
                x1, y1, x2, y2 = bbox

                frame = cv2.rectangle(data, (x1, y1), (x2, y2), (0, 255, 0), 5)

            cv2.imshow('frame', data)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    print('-----KeyboardInterrupt')
finally:
    cv2.destroyAllWindows()



cap = cv2.VideoCapture(2)
while True:
    ret, frame = cap.read()

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lowerLimit, upperLimit = get_limits(color=yellow)

    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

    mask_ = Image.fromarray(mask)

    bbox = mask_.getbbox()

    if bbox is not None:
        x1, y1, x2, y2 = bbox

        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()

