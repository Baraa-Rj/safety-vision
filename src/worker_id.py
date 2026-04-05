import cv2


class WorkerIdentifier:
    def __init__(self):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict)

    def identify(self, person_crop):
        """
        Takes a cropped image of a person.
        Returns worker ID (int) or None if no marker found.
        """
        gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
        _, ids, _ = self.aruco_detector.detectMarkers(gray)
        if ids is not None and len(ids) > 0:
            return int(ids[0][0])
        return None