import cv2
from classes import PanelDetector

p_det = PanelDetector.PanelDetector()
p_det.run()

if p_det.calibrate():
    while 1:
        frame = p_det.get_frame()
        contours, _ = p_det.get_contours(frame)

        p_det.remove_roi(contours)

        p_det.draw_roi(frame)

        p_det.draw_contours(frame, contours)

        p_det.draw_centroids(frame)

        topview = p_det.get_top_view(frame)

        # panel.draw_image(frame, 'BGR')
        p_det.draw_image(topview, 'Top View')

        # Waiting for interrupt
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break