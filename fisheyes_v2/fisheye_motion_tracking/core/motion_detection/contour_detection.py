import cv2

def detect_motion_regions(mask, min_area=500):

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    regions = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        cx = x + w // 2
        cy = y + h // 2

        regions.append({
            "bbox": (x, y, w, h),
            "center": (cx, cy),
            "area": area
        })

    return regions