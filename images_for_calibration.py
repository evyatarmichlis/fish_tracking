import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
def get_images(Itype = 'left_right'):
    cap = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)

    num = 0

    while cap.isOpened():

        succes1, img = cap.read()
        succes2, img2 = cap2.read()

        k = cv2.waitKey(5)

        if k == 27:
            break
        elif k == ord('s'): # wait for 's' key to save and exit
            if Itype =="left_right":
                cv2.imwrite(r'C:/Users/Evyatar/PycharmProjects/pythonProject14/images/stereoLeft/image' + str(num) + '.png', img)
                cv2.imwrite(r'C:/Users/Evyatar/PycharmProjects/pythonProject14/images/stereoRight/image' + str(num) + '.png', img2)
            elif Itype == "synched": # side fish
                cv2.imwrite(r'C:/Users/Evyatar/PycharmProjects/pythonProject14/images/synched/image' + str(num) + 'r.png',img)
                cv2.imwrite(r'C:/Users/Evyatar/PycharmProjects/pythonProject14/images/synched/image' + str(num) + 'l.png',img2)
            elif Itype =="test":
                cv2.imwrite(r'C:/Users/Evyatar/PycharmProjects/pythonProject14/images/test/image' + str(num) + 'r.png',img)
                cv2.imwrite(r'C:/Users/Evyatar/PycharmProjects/pythonProject14/images/test/image' + str(num) + 'l.png',img2)
            else:
                raise Exception("not a valid type")

            print("images saved!")
            num += 1
        # img = mpimg.imread(img)
        # imgplot = plt.imshow(img)
        # plt.show()
        cv2.imshow('Img 1',img)
        cv2.imshow('Img 2',img2)

    # Release and destroy all windows before termination
    cap.release()
    cap2.release()

    cv2.destroyAllWindows()
get_images()
#
#
# pygame.camera.init()
#
# cameras = pygame.camera.list_cameras()
#
# if not cameras:
#     raise ValueError("No cameras found")
#
# camera = pygame.camera.Camera(cameras[0])
# camera.start()
# while True:
#     image = camera.get_image()