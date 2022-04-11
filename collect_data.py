import cv2
import time

cap = cv2.VideoCapture(0)
idx = 0
ms = 0.1
t = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        break

    cv2.imshow('img', frame)


    if time.time() - t >= ms:
        cv2.imwrite('ae_images/'+str(idx)+'.jpg', frame)
        t = time.time()
        idx += 1