import numpy
import cv2
from math import isclose


class PanelDetector:
    # Constructor
    def __init__(self, index=0, calibrations=10):
        self.__debug = True
        # initialize video device
        self.__capture = cv2.VideoCapture(index)
        # define number of calibration
        self.__calibrations = calibrations
        # epsilon for center of buttons comparison in pixels
        self.__distance = 10
        # epsilon for rectangle detection in percent
        self.__curvy = 0.04
        # epsilon for sector detection in pixels
        self.__sectorepsilon = 100
        # is the sectors found or not
        self.__sectors_found = False
        # is the largest rectangle found or not
        self.__roi_found = False
        # largest contour
        self.__roi_contour = None
        # rectangle of interest where panel will be found
        self.__roi = (0, 0, 0, 0)
        # homography matrix
        self.__homography = numpy.empty(shape=(3, 3))
        # corners of the panel
        self.__corners = numpy.empty(shape=(4, 2))
        # number of the panel's buttons
        self.__nbuttons = 12
        # centers of the panel's buttons
        self.__centroids = numpy.array([], dtype=numpy.uint8)
        # corners of the top view of panel
        self.__top_view_centroids = numpy.array([], dtype=numpy.uint8)
        # kernel matrices for morphological transformation
        self.__kernel_square = numpy.ones((5, 5), numpy.uint8)
        self.__kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

        self.__fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=10, detectShadows=False)

        # Template frame
        self.__template = None
        # Previous frame
        self.__prev = None
        # Current frame
        self.__curr = None
        # Next frame
        self.__next = None

    # Destructor
    def __delete__(self, instance):
        self.__capture.release()
        cv2.destroyAllWindows()

    # Distance between two points in pixels
    def p_metric(self, a, b):
        return numpy.sqrt(numpy.sum((numpy.array(a) - numpy.array(b))**2))

    # Distance between two colors
    def c_metric(self, a, b):
        return numpy.sum((a-b)**2)

    # Difference between areas of two contours in pixels
    def is_equal_contours(self, a, b):
        return isclose(cv2.contourArea(a), cv2.contourArea(b), abs_tol=cv2.contourArea(b) * 0.1)

    # Difference between areas of two contours in pixels
    def area_metric(self, a, b):
        return cv2.contourArea(a) - cv2.contourArea(b)

    # Return centroids of sectors
    def get_centroids(self):
        return self.__centroids

    # Return homography
    def get_homography(self):
        return self.__homography

    # Get current frame from camera
    def get_frame(self):
        # Read image from camera
        retval, frame = self.__capture.read()
        if retval:
            # Perform blur to achieve better quality
            return cv2.blur(frame, (5, 5))
        else:
            raise Exception("Can't get frame!")

    # Remove unnecessary area
    def pick_out_roi(self, bit):
        mask = numpy.zeros(bit.shape, numpy.uint8)
        if self.__roi_found:
            x, y, w, h = self.__roi
        else:
            x, y, w, h = 0, bit.shape[0] // 2, bit.shape[1], bit.shape[0] // 2
        mask[y:y + h, x:x + w] = bit[y:y + h, x:x + w]
        return mask

    # Initialize next frame
    def next_frame(self):
        bgr = self.get_frame()

        # Read next image
        self.__prev = self.__curr
        self.__curr = self.__next
        gray = cv2.cvtColor(bgr, cv2.COLOR_RGB2GRAY)
        # gray = self.substract_bg(bgr)

        # Perform morphological transformations to filter out the background noise
        gray = cv2.erode(gray, self.__kernel_square, iterations=1)
        gray = cv2.dilate(gray, self.__kernel_ellipse, iterations=1)

        # self.__next = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
        self.__next = cv2.adaptiveThreshold(gray, 127, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)

        return bgr

    # Get image with motion
    def get_motion(self, image):
        self.next_frame()
        # return cv2.bitwise_and(cv2.absdiff(self.__next, self.__curr), cv2.absdiff(self.__curr, self.__prev))
        # return self.substract_bg(image)
        return cv2.threshold(numpy.array(numpy.sqrt(numpy.sum((self.pick_out_roi(self.__template - image))**2, axis=-1)), dtype=numpy.uint8), 22, 255, cv2.THRESH_BINARY)[1]

    # Substract background
    def substract_bg(self, frame):
        return self.__fgbg.apply(frame)

    # Get contours from current BGR frame
    def get_contours(self, frame):
        # Transform BGR frame to Gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform Canny edges detector and getting a binary image
        bit = cv2.Canny(gray, 0, 40, apertureSize=3)

        # Perform morphological transformations to filter out the background noise
        bit = cv2.dilate(bit, self.__kernel_ellipse, iterations=2)
        bit = cv2.erode(bit, self.__kernel_square, iterations=1)
        # cv2.morphologyEx(bit, cv2.MORPH_CLOSE, )

        bit = self.pick_out_roi(bit)

        # Looking for all contours in binary image
        _, contours, hierarchy = cv2.findContours(bit, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if self.__debug:
            self.draw_image(bit, 'Binary')

        return contours, hierarchy

    # Draw contours on current BGR frame
    def draw_contours(self, image, contours, color=(0, 0, 255)):
        for c in contours:
            if len(cv2.approxPolyDP(c, self.__curvy * cv2.arcLength(c, True), True)) == 4:
                cv2.drawContours(image, [c], 0, color, -1)
        return image

    # Draw image
    def draw_image(self, image, window_name='default'):
        cv2.imshow(window_name, image)

    # Draw top view of panel
    def get_top_view(self, image):
        return cv2.warpPerspective(image, self.__homography, (image.shape[1], image.shape[0]))

    # Find contour with maximum area
    def get_max_contour(self, contours, hierarchy=None):
        if hierarchy is None:
            return max(contours, key=cv2.contourArea)
        else:
            # TODO Implement getting max contour using hierarchy
            pass

    # Find center of mass of contour
    def centroid(self, contour):
        m = cv2.moments(contour)
        return int(m["m10"]/m["m00"]), int(m["m01"]/m["m00"])

    # Draw center of mass of contour
    def draw_centroids(self, frame, color=(0, 255, 0)):
        for b in self.__centroids:
            cv2.circle(frame, b, 5, color, 5)
        return frame

    # Draw center of mass of contour
    def draw_top_view_centroids(self, frame, color=(0, 255, 0)):
        for b in self.__top_view_centroids:
            cv2.circle(frame, b, 5, color, 5)
        return frame

    # Remove rectangle of interest to see sectors
    def remove_roi(self, contours):
        if self.__roi_found:
            # Looking for the largest contour
            max_contour = self.get_max_contour(contours)

            # Removing largest contour from all contours
            if self.is_equal_contours(max_contour, self.__roi_contour):
                contours.remove(max_contour)
        return contours

    # Define rectangle of interest
    def roi(self, contour):
        self.__roi_contour = contour
        self.__roi = cv2.boundingRect(contour)
        approx = cv2.approxPolyDP(contour, self.__curvy * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            self.__corners = [tuple(c) for c in approx.reshape(-1, 2)]
            # Calculate Homography for 8 points of perspective projection
            self.__homography, _ = cv2.findHomography(numpy.array(self.__corners),
                                                      numpy.array([[0, 720], [1280, 720], [1280, 0], [0, 0]]))

        else:
            # TODO To think about second way
            pass

    # Draw rectangle of interest
    def draw_roi(self, frame, color=(0, 255, 0)):
        for i in range(1, len(self.__corners)):
            cv2.line(frame, self.__corners[i-1], self.__corners[i], color, 2)
        return cv2.line(frame, self.__corners[0], self.__corners[len(self.__corners)-1], color, 2)

    # Find rectangle of interest where panel is located
    def find_roi(self):
        for t in range(0, self.__calibrations):

            # Get current frame from camera
            frame = self.get_frame()

            # Get contours from current frame
            contours, hierarchy = self.get_contours(frame)

            # TODO To think about other ways to avoid empty array
            if len(contours) == 0:
                continue

            # Looking for the largest contour
            max_contour = self.get_max_contour(contours)

            # If roi is not
            if self.__roi_contour is None:
                self.roi(max_contour)
            else:
                if not self.__roi_found:
                    # If contour area and ordinate of new contour is more than largest contour then redefine it
                    if self.area_metric(max_contour, self.__roi_contour) > 0 and \
                                    self.centroid(max_contour)[1] < self.centroid(self.__roi_contour)[1]:
                        self.roi(max_contour)

            if self.__debug:
                x, y, w, h = self.__roi
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                self.draw_image(self.draw_contours(frame, contours), 'BGR')

                # Waiting for interrupt
                k = cv2.waitKey(5) & 0xFF
                if k == 27:
                    break

            # if largest contour found
            if len(self.__roi_contour) > 0 and not self.__roi_found:
                self.__roi_found = True
                print("Largest contour found!")
                return True

        return False

    # Find centroids of sectors where buttons are located
    def find_sectors(self):
        for t in range(0, self.__calibrations):

            # Get current frame from camera
            frame = self.get_frame()

            # Get contours from current frame
            contours, hierarchy = self.get_contours(frame)

            # TODO To think about other ways to avoid empty array
            if len(contours) == 0:
                continue

            self.remove_roi(contours)

            # TODO It is bottleneck
            if sum(cv2.contourArea(c) >= self.__sectorepsilon for c in contours) == 8:
                button_contours = [c for c in contours if cv2.contourArea(c) >= self.__sectorepsilon]
                self.__centroids = [self.centroid(c) for c in button_contours]
                self.__top_view_centroids = [(int(x//z), int(y//z))
                                             for c in self.__centroids for x, y, z in
                                             [numpy.dot(self.__homography, numpy.append(numpy.array(c), [1]))]]
                self.__sectors_found = True
                print("Sectors found!")
                return True

            if self.__debug:
                x, y, w, h = self.__roi
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                self.draw_image(self.draw_contours(frame, contours), 'BGR')

                # Waiting for interrupt
                k = cv2.waitKey(5) & 0xFF
                if k == 27:
                    break

        return False

    # Start calibration
    def calibrate(self):

        # Firstly looking for rectangle of interest where panel is located
        if self.find_roi():
            # Then looking for sectors where buttons are located
            if self.find_sectors():
                for i in range(0, 3):
                    self.__template = self.get_frame()
                    self.next_frame()
                print('Calibration successful!')
                return True
            else:
                return False
        else:
            return False

    # Run imitation
    def run(self):

        if self.calibrate():
            while 1:
                frame = self.next_frame()
                contours, _ = self.get_contours(frame)

                move = self.get_motion(frame)
                cv2.normalize(move, move, 0, 255, cv2.NORM_MINMAX)

                self.remove_roi(contours)

                if self.__debug:
                    self.draw_roi(frame)

                    self.draw_contours(frame, contours)

                    topview = self.get_top_view(frame)

                    self.draw_centroids(frame)
                    self.draw_top_view_centroids(topview)

                    self.draw_image(frame, 'BGR')
                    self.draw_image(topview, 'Top View')
                    self.draw_image(move, 'Movement indicator')

                # Waiting for interrupt
                k = cv2.waitKey(5) & 0xFF
                if k == 27:
                    break
        else:
            return False
