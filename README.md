"""
VehicleDetector: pure classical Computer Vision pipeline.

Pipeline:
  Frame -> MOG2 background subtraction -> threshold (buang shadow)
        -> morphological opening (buang noise) -> closing (tutup lubang)
        -> findContours -> filter by area -> bounding box + centroid

Counting:
  pointPolygonTest untuk cek centroid kendaraan di dalam polygon ROI.

NOT a machine learning / deep learning model.
"""

import cv2
import numpy as np


class VehicleDetector:
    def __init__(
        self,
        history: int = 500,
        threshold: int = 16,
        detect_shadows: bool = True,
        kernel_size: int = 5,
        open_iter: int = 2,
        close_iter: int = 2,
    ):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=threshold,
            detectShadows=detect_shadows,
        )
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        self.open_iter = open_iter
        self.close_iter = close_iter

    def get_foreground_mask(self, frame: np.ndarray) -> np.ndarray:
        """Hasilkan binary mask foreground (kendaraan bergerak)."""
        fg_mask = self.bg_subtractor.apply(frame, learningRate=0.0005)

        # MOG2 menandai shadow sebagai value 127 (abu-abu).
        # Threshold 200 -> hanya pixel "definitely foreground" yang dipertahankan.
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Opening: erosion lalu dilation -> hilangkan noise titik kecil
        fg_mask = cv2.morphologyEx(
            fg_mask, cv2.MORPH_OPEN, self.kernel, iterations=self.open_iter
        )
        # Closing: dilation lalu erosion -> tutup lubang dalam blob kendaraan
        fg_mask = cv2.morphologyEx(
            fg_mask, cv2.MORPH_CLOSE, self.kernel, iterations=self.close_iter
        )
        return fg_mask

    def find_vehicles(
        self,
        fg_mask: np.ndarray,
        min_area: int = 500,
        max_area: int = 50000,
    ) -> list:
        """
        Cari kontur dari foreground mask, filter by area,
        return list of dict {centroid, bbox, area}.
        """
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        vehicles = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (min_area <= area <= max_area):
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2
            vehicles.append(
                {"centroid": (cx, cy), "bbox": (x, y, w, h), "area": float(area)}
            )
        return vehicles

    @staticmethod
    def count_in_roi(vehicles: list, roi_polygon: list) -> tuple:
        """
        Hitung kendaraan yang centroid-nya berada di dalam polygon ROI.
        Return (count, list_of_vehicles_in_roi).
        """
        polygon = np.array(roi_polygon, dtype=np.int32)
        in_roi = []
        for v in vehicles:
            # pointPolygonTest: >= 0 berarti di dalam atau di garis polygon
            if cv2.pointPolygonTest(polygon, v["centroid"], False) >= 0:
                in_roi.append(v)
        return len(in_roi), in_roi

    # SINGLE-IMAGE DETECTION (untuk Image Mode, tanpa MOG2)
    @staticmethod
    def detect_from_image(
        image: np.ndarray,
        saturation_thresh: int = 45,
        bright_thresh: int = 175,
        dark_thresh: int = 55,
        kernel_size: int = 5,
        open_iter: int = 1,
        close_iter: int = 2,
    ) -> np.ndarray:
        """
        Deteksi kendaraan dari 1 frame statis dengan HSV color analysis.

        Pipeline (pure classical CV, no ML/DL):
          1. Convert BGR → HSV color space
          2. Identifikasi pixel "bukan-jalan":
             - Saturation tinggi → kendaraan berwarna (bus kuning, truk merah, dll)
             - Brightness sangat tinggi → mobil putih, atau marka jalan (difilter di
               find_vehicles by aspect ratio)
             - Brightness sangat rendah → mobil hitam
          3. Morphological opening (hapus noise) + closing (tutup lubang)

        Kenapa HSV color-based lebih cocok untuk drone view:
          - Tidak terganggu bayangan (shadow tetap grey, bukan saturated)
          - Mobil yang bersebelahan TIDAK merge jadi 1 blob (tiap mobil tetap punya
            warna distinct)
          - Konsisten di berbagai kondisi cahaya
          - Edge-based (Canny) gagal karena: roof mobil dari atas itu uniform color,
            edge antar mobil tidak kuat, dan dilation menggabungkan beberapa mobil
            jadi 1 blob besar yang sering melebihi max_area.

        Args:
            image: BGR input image
            saturation_thresh: Pixel dengan saturation > nilai ini = berwarna.
                Lebih kecil = lebih sensitif (catch mobil pucat juga).
            bright_thresh: Pixel dengan value > ini = sangat terang.
                Lebih besar = lebih strict (hanya mobil putih murni).
            dark_thresh: Pixel dengan value < ini = sangat gelap.
                Lebih kecil = lebih strict (hanya mobil hitam pekat).
            kernel_size: Ukuran kernel ellipse untuk morphology.
            open_iter: Iterasi opening untuk hapus noise.
            close_iter: Iterasi closing untuk tutup lubang dalam blob mobil.

        Returns:
            Binary mask 8-bit (putih = kandidat kendaraan).
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        # Non-road pixel = saturated OR very bright OR very dark
        mask = (
            (saturation > saturation_thresh) |
            (value > bright_thresh) |
            (value < dark_thresh)
        ).astype(np.uint8) * 255

        # Morphological cleanup
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iter)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)

        return mask

    @staticmethod
    def process_single_image(
        image: np.ndarray,
        rois: dict,
        saturation_thresh: int = 45,
        bright_thresh: int = 175,
        dark_thresh: int = 55,
        kernel_size: int = 5,
        open_iter: int = 1,
        close_iter: int = 2,
        min_area: int = 400,
        max_area: int = 50000,
        max_aspect_ratio: float = 6.0,
    ) -> dict:
        """
        End-to-end single-image detection: HSV color → count per ROI.

        Filter contour dengan:
          - Area: min_area <= area <= max_area
          - Aspect ratio: max(w,h)/min(w,h) <= max_aspect_ratio
            → menolak lane marking yang panjang & tipis (biasanya aspect >10)

        Returns dict {
            'mask': binary detection mask,
            'all_vehicles': list semua kendaraan terdeteksi,
            'vehicles_per_lane': dict {lane: list},
            'counts': dict {lane: count}
        }
        """
        # Detect mask
        mask = VehicleDetector.detect_from_image(
            image,
            saturation_thresh=saturation_thresh,
            bright_thresh=bright_thresh,
            dark_thresh=dark_thresh,
            kernel_size=kernel_size,
            open_iter=open_iter,
            close_iter=close_iter,
        )

        # Find contours, filter by area AND aspect ratio
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        all_vehicles = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (min_area <= area <= max_area):
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            # Filter lane markings: aspect ratio terlalu ekstrim = bentuk panjang/tipis
            aspect = max(w, h) / max(1, min(w, h))
            if aspect > max_aspect_ratio:
                continue
            all_vehicles.append(
                {"centroid": (x + w // 2, y + h // 2),
                 "bbox": (x, y, w, h),
                 "area": float(area)}
            )

        # Count per ROI (point-in-polygon test)
        vehicles_per_lane = {}
        counts = {}
        for lane_name, polygon in rois.items():
            count, in_roi = VehicleDetector.count_in_roi(all_vehicles, polygon)
            vehicles_per_lane[lane_name] = in_roi
            counts[lane_name] = count

        return {
            "mask": mask,
            "all_vehicles": all_vehicles,
            "vehicles_per_lane": vehicles_per_lane,
            "counts": counts,
        }

    def process(
        self,
        frame: np.ndarray,
        rois: dict,
        min_area: int = 500,
        max_area: int = 50000,
    ) -> dict:
        """
        End-to-end satu frame.

        Args:
            frame: BGR frame dari video.
            rois: dict {lane_name: polygon_points}.
            min_area, max_area: filter area kontur.

        Returns:
            dict {
                'fg_mask': binary mask,
                'all_vehicles': list semua kendaraan terdeteksi,
                'vehicles_per_lane': dict {lane: list vehicles in lane},
                'counts': dict {lane: count}
            }
        """
        fg_mask = self.get_foreground_mask(frame)
        all_vehicles = self.find_vehicles(fg_mask, min_area=min_area, max_area=max_area)

        vehicles_per_lane = {}
        counts = {}
        for lane_name, polygon in rois.items():
            count, in_roi = self.count_in_roi(all_vehicles, polygon)
            vehicles_per_lane[lane_name] = in_roi
            counts[lane_name] = count

        return {
            "fg_mask": fg_mask,
            "all_vehicles": all_vehicles,
            "vehicles_per_lane": vehicles_per_lane,
            "counts": counts,
        }