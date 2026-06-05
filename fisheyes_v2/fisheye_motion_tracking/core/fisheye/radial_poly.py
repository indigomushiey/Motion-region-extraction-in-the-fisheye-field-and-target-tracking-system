import json
import cv2
import numpy as np

from pathlib import Path


# =========================
# Project Root
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "homework2"

CALIB_DIR = DATA_DIR / "calibration_data"
IMAGE_DIR = DATA_DIR / "rgb_images"


class RadialPolyCamera:
    def __init__(self, json_path):

        with open(json_path, 'r') as f:
            data = json.load(f)

        intrinsic = data["intrinsic"]

        self.width = int(intrinsic["width"])
        self.height = int(intrinsic["height"])

        self.cx = self.width / 2 + intrinsic["cx_offset"]
        self.cy = self.height / 2 + intrinsic["cy_offset"]

        self.k1 = intrinsic["k1"]
        self.k2 = intrinsic["k2"]
        self.k3 = intrinsic["k3"]
        self.k4 = intrinsic["k4"]

    def undistort(self, image, fov_scale=0.5):

        h, w = image.shape[:2]

        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)

        f = self.k1 * fov_scale

        for y in range(h):
            for x in range(w):

                xn = (x - self.cx) / f
                yn = (y - self.cy) / f

                r_u = np.sqrt(xn * xn + yn * yn)

                theta = np.arctan(r_u)

                theta2 = theta * theta
                theta3 = theta2 * theta
                theta5 = theta3 * theta2
                theta7 = theta5 * theta2

                r_d = (
                    self.k1 * theta
                    + self.k2 * theta3
                    + self.k3 * theta5
                    + self.k4 * theta7
                )

                if r_u > 1e-8:
                    scale = r_d / r_u
                else:
                    scale = 1.0

                x_distorted = self.cx + xn * scale
                y_distorted = self.cy + yn * scale

                map_x[y, x] = x_distorted
                map_y[y, x] = y_distorted

        undistorted = cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR
        )

        return undistorted


if __name__ == "__main__":

    frame_id = "00004_FV"

    json_path = CALIB_DIR / f"{frame_id}.json"
    image_path = IMAGE_DIR / f"{frame_id}.png"

    camera = RadialPolyCamera(json_path)

    image = cv2.imread(str(image_path))

    result = camera.undistort(
        image,
        fov_scale = 1.2
    )

    cv2.imshow("original", image)
    cv2.imshow("undistorted", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
