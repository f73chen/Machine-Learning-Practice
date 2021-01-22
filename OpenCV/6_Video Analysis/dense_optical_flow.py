# get 2-channel array with optical flow vectors (u, v)
    # find mag and direction
# colour code results for visualization where:
    # direction = hue
    # magitude = value

import cv2
import numpy as np

cap = cv2.VideoCapture('slow.flv')

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

while True:
    ret, frame2 = cap.read()
    nexti = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # input1: prev --> first 8-bit single-channel input image
    # input2: next --> second image of same size and type as prev
    # input3: flow --> computed flow image, same size as prev and type CV_32FC2
    # input4: pyr_scale --> image scale (<1) to build pyramids for each image
    # input5: levels --> number of pyramid layers including initial image
    # input6: winsize --> average window size
        # larger = more robust to noise & tracks bigger movements
        # however yield more blurred motion field
    # input7: iterations --> number of iterations at each pyramid level
    # input8: poly_n --> size of pixel neighbourhood to find polynomial expansion in each pixel
        # larger = image approximated with smoother surfaces
        # more robust but more blurred motion field
    # input9: poly_sigma --> std of Gaussian used to smooth derivatives
        # basis for polynomial expansion
    # input10: flags --> operation flags, like:
        # OPTFLOW_USE_INITIAL_FLOW --> use input flow as initial flow approx
        # OPTFLOW_FARNEBACK_GAUSSIAN --> use Gaussian winsize * winsize filter instead of box filter. More accurate but slower
    # output1: flow --> optical flow for each prev pixel
    flow = cv2.calcOpticalFlowFarneback(prvs, nexti, None, 0.5, 3, 50, 3, 10, 1.5, 0)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow('frame2', rgb)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    prvs = nexti

cap.release()
cv2.destroyAllWindows()