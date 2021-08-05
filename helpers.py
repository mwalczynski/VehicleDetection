import numpy as np
import cv2


CONFIDENCE_THRESHOLD = 0.3


def mapPrediction(prediction):
    top_x = prediction["topleft"]["x"]
    top_y = prediction["topleft"]["y"]

    btm_x = prediction["bottomright"]["x"]
    btm_y = prediction["bottomright"]["y"]

    confidence = round(prediction["confidence"], 3)
    label = prediction["label"]  # + " " + str(round(confidence, 3))

    return (top_x, top_y, btm_x, btm_y, confidence, label, 0)


def getRectangleCenter(top_x, top_y, btm_x, btm_y):
    centerCoord = ((top_x + btm_x) / 2, (top_y + btm_y) / 2)
    return centerCoord


def getDistance(point1, point2):
    dx = point1[0] - point2[0]
    dy = point1[1] - point2[1]
    dsquared = dx ** 2 + dy ** 2
    result = dsquared ** 0.5
    return result


def getDistanceBetweenRectangleCenters(
    top_x1, top_y1, btm_x1, btm_y1, top_x2, top_y2, btm_x2, btm_y2
):
    center1 = getRectangleCenter(top_x1, top_y1, btm_x1, btm_y1)
    center2 = getRectangleCenter(top_x2, top_y2, btm_x2, btm_y2)
    distance = getDistance(center1, center2)
    return distance


def getDistanceBetweenCarAndTaxi(car, taxi):
    top_x1, top_y1, btm_x1, btm_y1, _, _, _ = car
    top_x2, top_y2, btm_x2, btm_y2, _, _, _ = taxi
    carCenter = getRectangleCenter(top_x1, top_y1, btm_x1, top_y1)
    taxiCenter = getRectangleCenter(top_x2, top_y2, btm_x2, btm_y2)
    distance = getDistance(carCenter, taxiCenter)
    return distance


def mapCarsToTaxis(objects):
    taxis = list(filter(lambda x: x[5] == "Taxi", objects))
    notTaxis = list(filter(lambda x: x[5] != "Taxi", objects))
    results = []

    for element in notTaxis:
        top_x, top_y, btm_x, btm_y, confidence, label, index = element

        if label == "Autobus":
            results.append(element)

        if label == "Car":
            isTaxi = False
            for taxi in taxis:
                distance = getDistanceBetweenCarAndTaxi(element, taxi)
                if distance <= 100 and distance > 0:
                    toAdd = (top_x, top_y, btm_x, btm_y, confidence, "Taxi", index)
                    results.append(toAdd)
                    isTaxi = True
                    break
            if isTaxi is False:
                results.append(element)

    return results


def boxing(original_img, predictions):
    newImage = np.copy(original_img)

    predictions = list(filter(lambda x: x[4] >= CONFIDENCE_THRESHOLD, predictions))

    mappedResults = mapCarsToTaxis(predictions)

    for result in mappedResults:
        top_x, top_y, btm_x, btm_y, confidence, label, index = result

        if label == "Car":
            newImage = cv2.rectangle(
                newImage, (top_x, top_y), (btm_x, btm_y), (0, 0, 255), 3
            )
            newImage = cv2.putText(
                newImage,
                label + " " + str(index),
                (top_x, top_y - 5),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                0.8,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        if label == "Taxi":
            newImage = cv2.rectangle(
                newImage, (top_x, top_y), (btm_x, btm_y), (0, 255, 0), 3
            )
            newImage = cv2.putText(
                newImage,
                label + " " + str(index),
                (top_x, top_y - 5),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                0.8,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        if label == "Autobus":
            newImage = cv2.rectangle(
                newImage, (top_x, top_y), (btm_x, btm_y), (0, 255, 0), 3
            )
            newImage = cv2.putText(
                newImage,
                label + " " + str(index),
                (int((top_x + btm_x) / 2), int((top_y + btm_y) / 2)),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                0.8,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    return newImage


def mapTrackToPred(tracks, pred):
    new_pred = []
    for p in pred:
        track = min(
            tracks,
            key=lambda t: getDistanceBetweenRectangleCenters(
                t[0], t[1], t[2], t[3], p[0], p[1], p[2], p[3]
            ),
        )
        p = (p[0], p[1], p[2], p[3], p[4], p[5], int(track[4]))
        new_pred.append(p)
    return new_pred
