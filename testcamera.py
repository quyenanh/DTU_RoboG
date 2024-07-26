from ugot import ugot
import cv2
import numpy as np
(w, h) = (640,240)  # Resolution
got = ugot.UGOT()

got.initialize('10.220.5.228')

got.open_camera()
no_points_count = 0
try:
    while True:
        frame = got.read_camera_data()
        if frame is not None:

            nparr = np.frombuffer(frame, np.uint8)
            data = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
            data.shape = (h, w)

            cv2.imshow("frame", data)
            cv2.waitKey(1)
except KeyboardInterrupt:
    print('-----KeyboardInterrupt')
