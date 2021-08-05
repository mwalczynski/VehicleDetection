import matplotlib.pyplot as plt
import numpy as np

from darkflow.net.build import TFNet
import cv2

from sort import *
from helpers import *

options = {"model": "cfg/custom.cfg", "load": -2, "gpu": 1.0}

tfnet = TFNet(options)
tfnet.load_from_ckpt()


def image():
    imgcv = cv2.imread("./testImg.png")
    results = tfnet.return_predict(imgcv)
    mappedResults = list(
        filter(
            lambda x: x[4] >= CONFIDENCE_THRESHOLD,
            map(lambda x: mapPrediction(x), results),
        )
    )
    new_frame = boxing(imgcv, mappedResults)
    while True:
        cv2.imshow("video", new_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def video():
    cap = cv2.VideoCapture("./testVid.avi")

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    out = cv2.VideoWriter("./classified.avi", fourcc, 30.0, (int(width), int(height)))

    counter = 0

    line_bottom_y = 285

    previous_side = [None] * 1000
    already_counter = [None] * 1000
    for i in range(0, 1000):
        already_counter[i] = False
        previous_side[i] = 0

    mot_tracker = Sort()
    while True:
        # Read a new frame
        ok, frame = cap.read()
        for i in range(0, 5):
            ok, frame = cap.read()
        if not ok:
            break

        results = tfnet.return_predict(frame)
        mappedResults = list(
            filter(
                lambda x: x[4] >= CONFIDENCE_THRESHOLD,
                map(lambda x: mapPrediction(x), results),
            )
        )
        objectsToTrack = list(
            map(
                lambda x: [x[0], x[1], x[0] + x[2] - x[0], x[1] + x[3] - x[1], x[4]],
                mappedResults,
            )
        )
        if objectsToTrack:
            objectsToTrack = np.asarray(objectsToTrack)
            track_bbs_ids = mot_tracker.update(objectsToTrack)
            if track_bbs_ids.any():
                mappedResults = mapTrackToPred(track_bbs_ids, mappedResults)

        current = list(
            filter(
                lambda x: x[3] != "Taxi",
                map(lambda x: (x[1], x[3], x[6], x[5]), mappedResults),
            )  # top_y, bot_y, index, label
        )

        for cur in current:
            cur_index = cur[2]
            cur_side = line_bottom_y - (cur[0] + cur[1]) / 2
            if (
                already_counter[cur_index] is False
                and previous_side[cur_index] * cur_side < 0
            ):
                already_counter[cur_index] = True
                counter += 1
            else:
                previous_side[cur_index] = cur_side

        cv2.line(frame, (350, line_bottom_y), (870, line_bottom_y), (0, 255, 255), 3)
        cv2.putText(
            frame,
            str(counter),
            (100, 60),
            cv2.FONT_HERSHEY_DUPLEX,
            2.0,
            (0, 255, 255),
            6,
        )

        new_frame = boxing(frame, mappedResults)
        out.write(new_frame)

        cv2.imshow("video", new_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


video()
