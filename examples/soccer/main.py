import argparse
import pandas as pd
from enum import Enum
from typing import Iterator, List

import os
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.common.ball import BallTracker, BallAnnotator
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-player-detection.pt')
PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-pitch-detection.pt')
BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-ball-detection.pt')

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

STRIDE = 60
CONFIG = SoccerPitchConfiguration()

COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']
VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
    color=[sv.Color.from_hex(color) for color in CONFIG.colors],
    text_color=sv.Color.from_hex('#FFFFFF'),
    border_radius=5,
    text_thickness=1,
    text_scale=0.5,
    text_padding=5,
)
EDGE_ANNOTATOR = sv.EdgeAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    thickness=2,
    edges=CONFIG.edges,
)
TRIANGLE_ANNOTATOR = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    base=20,
    height=15,
)
BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)


class Mode(Enum):
    """
    Enum class representing different modes of operation for Soccer AI video analysis.
    """
    PITCH_DETECTION = 'PITCH_DETECTION'
    PLAYER_DETECTION = 'PLAYER_DETECTION'
    BALL_DETECTION = 'BALL_DETECTION'
    PLAYER_TRACKING = 'PLAYER_TRACKING'
    TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'
    RADAR = 'RADAR'
    EXTRACT = 'EXTRACT'


def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    """
    Extract crops from the frame based on detected bounding boxes.

    Args:
        frame (np.ndarray): The frame from which to extract crops.
        detections (sv.Detections): Detected objects with bounding boxes.

    Returns:
        List[np.ndarray]: List of cropped images.
    """
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.array,
    goalkeepers: sv.Detections
) -> np.ndarray:
    """
    Resolve the team IDs for detected goalkeepers based on the proximity to team
    centroids.

    Args:
        players (sv.Detections): Detections of all players.
        players_team_id (np.array): Array containing team IDs of detected players.
        goalkeepers (sv.Detections): Detections of goalkeepers.

    Returns:
        np.ndarray: Array containing team IDs for the detected goalkeepers.

    This function calculates the centroids of the two teams based on the positions of
    the players. Then, it assigns each goalkeeper to the nearest team's centroid by
    calculating the distance between each goalkeeper and the centroids of the two teams.
    """
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id)


def render_radar(
    detections: sv.Detections,
    keypoints: sv.KeyPoints,
    color_lookup: np.ndarray,
    possession_id: int = None
) -> np.ndarray:
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    transformer = ViewTransformer(
        source=keypoints.xy[0][mask].astype(np.float32),
        target=np.array(CONFIG.vertices)[mask].astype(np.float32)
    )
    xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    transformed_xy = transformer.transform_points(points=xy)

    radar = draw_pitch(config=CONFIG)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 0],
        face_color=sv.Color.from_hex(COLORS[0]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 1],
        face_color=sv.Color.from_hex(COLORS[1]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 2],
        face_color=sv.Color.from_hex(COLORS[2]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 3],
        face_color=sv.Color.from_hex(COLORS[3]), radius=20, pitch=radar)
    return radar


def run_pitch_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run pitch detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """     
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(
            annotated_frame, keypoints, CONFIG.labels)
        yield annotated_frame


def run_player_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run player detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(annotated_frame, detections)
        yield annotated_frame


def run_ball_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run ball detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(
        callback=callback,
        overlap_filter_strategy=sv.OverlapFilter.NONE,
        slice_wh=(640, 640),
    )

    for frame in frame_generator:
        detections = slicer(frame).with_nms(threshold=0.1)
        detections = ball_tracker.update(detections)
        annotated_frame = frame.copy()
        annotated_frame = ball_annotator.annotate(annotated_frame, detections)
        yield annotated_frame


def run_player_tracking(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run player tracking on a video and yield annotated frames with tracked players.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels=labels)
        yield annotated_frame


def run_team_classification(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run team classification on a video and yield annotated frames with team colors.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
                players_team_id.tolist() +
                goalkeepers_team_id.tolist() +
                [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels, custom_color_lookup=color_lookup)
        yield annotated_frame

def calculate_ball_possession(players: sv.Detections, ball: sv.Detections, 
                              players_team_id: np.ndarray, method: str = 'euclidean') -> int:
    """
    Menentukan team_id mana yang menguasai bola berdasarkan metrik jarak yang dipilih.
    Mendukung: 'euclidean', 'manhattan', dan 'canberra'.
    """
    if len(ball) == 0 or len(players) == 0:
        return None
    
    if len(ball) > 1:
        ball = ball[0]

    player_coords = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    ball_coords = ball.get_anchors_coordinates(sv.Position.CENTER)

    # 4. Perhitungan Jarak Berdasarkan Metode (Referensi: Faisal dkk., 2020)
    if method == 'euclidean':
        distances = np.linalg.norm(player_coords - ball_coords, axis=1)
        threshold = 50  # Satuan cm/pixel
        
    elif method == 'manhattan':
        distances = np.sum(np.abs(player_coords - ball_coords), axis=1)
        threshold = 70  # Nilai lebih tinggi karena sifat penjumlahannya linear
        
    elif method == 'canberra':
        # sum(|xi - yi| / (|xi| + |yi|)) - Rasio Ternormalisasi
        num = np.abs(player_coords - ball_coords)
        den = np.abs(player_coords) + np.abs(ball_coords) + 1e-10
        # axis=1 menjumlahkan rasio sumbu X dan Y
        distances = np.sum(num / den, axis=1)
        threshold = 0.5  # Skala Canberra adalah 0-2 untuk data 2D
        
    else:
        raise ValueError("Metode tidak dikenal. Pilih: 'euclidean', 'manhattan', atau 'canberra'.")

    # 5. Cari Pemain Terdekat
    closest_player_index = np.argmin(distances)
    min_distance = distances[closest_player_index]

    # 6. Validasi Jarak terhadap Threshold
    if min_distance > threshold: 
        return None 
        
    # 7. Kembalikan team_id pemain pemenang
    return players_team_id[closest_player_index]

def save_tracking_results(tracking_data, output_path="tracking_data.csv"):
    """Menyimpan seluruh histori pergerakan ke CSV."""
    if tracking_data:
        df = pd.DataFrame(tracking_data)
        df.to_csv(output_path, index=False)
        print(f" Data tracking berhasil disimpan ke {output_path}")

def run_tracking_extraction(source_video_path: str, device: str, output_csv: str = "tracking_data.csv") -> None:
    """Fungsi khusus ekstraksi koordinat pemain, kiper, wasit, dan bola ke CSV tanpa visualisasi."""
    import pandas as pd
    # model initiate
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)

    # Team Classification 
    print("Analyze jersey color")
    frame_generator_samples = sv.get_video_frames_generator(source_path=source_video_path, stride=STRIDE)
    crops = []
    for frame in tqdm(frame_generator_samples, desc='Collecting team crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    # tracking prep
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    tracking_histories = []
    frame_idx = 0
    
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

    print(f"Start data tracking to {output_csv}...")
    for frame in tqdm(frame_generator, total=video_info.total_frames, desc="Processing Frames"):
        # Deteksi Objek

        pitch_res = pitch_detection_model(frame, verbose=False)[0]
        player_res = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        ball_res = ball_detection_model(frame, imgsz=640, verbose=False)[0]

        keypoints = sv.KeyPoints.from_ultralytics(pitch_res)
        raw_detections = sv.Detections.from_ultralytics(player_res)
        
        # Update Tracker (Hanya untuk pemain, kiper, wasit)
        non_ball_detections = raw_detections[raw_detections.class_id != BALL_CLASS_ID]
        tracked_detections = tracker.update_with_detections(non_ball_detections)
        
        # Deteksi Bola (Tanpa Tracker)
        ball_detections = sv.Detections.from_ultralytics(ball_res)

        # Pisahkan kelas untuk klasifikasi tim
        players = tracked_detections[tracked_detections.class_id == PLAYER_CLASS_ID]
        goalkeepers = tracked_detections[tracked_detections.class_id == GOALKEEPER_CLASS_ID]
        referees = tracked_detections[tracked_detections.class_id == REFEREE_CLASS_ID]

        # Prediksi Tim
        if len(players) > 0:
            players_team_id = team_classifier.predict(get_crops(frame, players))
        else:
            players_team_id = np.array([])

        if len(goalkeepers) > 0:
            goalkeepers_team_id = resolve_goalkeepers_team_id(players, players_team_id, goalkeepers)
        else:
            goalkeepers_team_id = np.array([])

        # --- SINKRONISASI MERGE & COLOR LOOKUP (Mencegah IndexError) ---
        # Kita merge dengan urutan: Player -> Goalkeeper -> Referee
        merged_detections = sv.Detections.merge([players, goalkeepers, referees])
        
        # Bangun color_lookup dengan urutan yang sama persis dengan merge
        ids_list = []
        ids_list.extend(players_team_id.tolist())
        ids_list.extend(goalkeepers_team_id.tolist())
        ids_list.extend([REFEREE_CLASS_ID] * len(referees))
        color_lookup = np.array(ids_list)

        player_confidences = merged_detections.confidence if merged_detections.confidence is not None else []
        ball_confidences = ball_detections.confidence if ball_detections.confidence is not None else []

        # pitch keypoint transformation
        mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
        if mask.sum() >= 4:
            transformer = ViewTransformer(
                source=keypoints.xy[0][mask].astype(np.float32),
                target=np.array(CONFIG.vertices)[mask].astype(np.float32)
            )

            # extract player, goalkeeper, referee data
            if len(merged_detections) > 0:
                xy_pixel = merged_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                xy_radar = transformer.transform_points(points=xy_pixel)

                for i in range(len(merged_detections)):
                    # save data if index true
                    if i < len(color_lookup):
                        tracking_histories.append({
                            'frame': frame_idx,
                            'player_id': merged_detections.tracker_id[i] if merged_detections.tracker_id is not None else -1,
                            'team_id': color_lookup[i],
                            'x_cm': xy_radar[i][0],
                            'y_cm': xy_radar[i][1],
                            'confidence': player_confidences[i]
                        })

            # Ekstrak Data Bola
            if len(ball_detections) > 0:
                ball_pixel = ball_detections.get_anchors_coordinates(sv.Position.CENTER)
                ball_radar = transformer.transform_points(points=ball_pixel)
                for i in range(len(ball_radar)):
                    tracking_histories.append({
                        'frame': frame_idx,
                        'player_id': 999, # ball unique id
                        'team_id': BALL_CLASS_ID,
                        'x_cm': ball_radar[i][0],
                        'y_cm': ball_radar[i][1],
                        'confidence': ball_confidences[i]
                    })
        
        frame_idx += 1

    # 5. Save Final Result
    save_tracking_results(tracking_histories, output_csv)



def run_radar(source_video_path: str, device: str, method: str = 'euclidean', output_csv: str = "possession_stats_radar.csv") -> Iterator[np.ndarray]:
    """
    Menjalankan deteksi radar dan kalkulasi possession secara real-time, 
    serta menyimpan histori statistik ke dalam file CSV.
    """
    # 1. Inisialisasi Model
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)

    # 2. Inisialisasi Variabel Statistik & Logging
    possession_records = [] 
    frame_idx = 0
    count_team_0 = 0  # Pink Team
    count_team_1 = 0  # Blue Team
    total_possession_frames = 0
    last_possession_team = None

    # 3. Fit Team Classifier (Proses Stride untuk Hemat Memori)
    frame_generator_samples = sv.get_video_frames_generator(source_path=source_video_path, stride=STRIDE)
    crops = []
    for frame in tqdm(frame_generator_samples, desc='Collecting team crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    # 4. Main Loop Pemrosesan Video
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    
    for frame in frame_generator:
        # Deteksi Lapangan (Keypoints)
        result_pitch = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result_pitch)
        
        # Deteksi Pemain & Tracking
        result_player = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result_player)
        detections = tracker.update_with_detections(detections)

        # Deteksi Bola
        result_ball = ball_detection_model(frame, imgsz=640, verbose=False)[0]
        ball_detections = sv.Detections.from_ultralytics(result_ball)

        # Klasifikasi Tim Pemain
        players = detections[detections.class_id == PLAYER_CLASS_ID]
        if len(players) > 0:
            players_team_id = team_classifier.predict(get_crops(frame, players))
        else:
            players_team_id = np.array([])

        # --- LOGIKA POSSESSION (SINKRON DENGAN VIDEO) ---
        current_ball_possession_team = calculate_ball_possession(
            players=players, 
            ball=ball_detections, 
            players_team_id=players_team_id,
            method=method
        )
        
        # Logika Persistence (Mengingat pemilik terakhir)
        if current_ball_possession_team is not None:
            last_possession_team = current_ball_possession_team

        # Akumulasi statistik per frame
        if last_possession_team == 0:
            count_team_0 += 1
            total_possession_frames += 1
            possession_label = "POSSESSION: PINK TEAM"
            text_color = (147, 20, 255) 
        elif last_possession_team == 1:
            count_team_1 += 1
            total_possession_frames += 1
            possession_label = "POSSESSION: BLUE TEAM"
            text_color = (255, 191, 0)
        else:
            possession_label = "POSSESSION: SEARCHING..."
            text_color = (255, 255, 255)

        annotated_frame = frame.copy()

        # Render Teks Persentase ke Video
        if total_possession_frames > 0:
            perc_0 = (count_team_0 / total_possession_frames) * 100
            perc_1 = (count_team_1 / total_possession_frames) * 100
            stat_text = f"PINK: {perc_0:.1f}% | BLUE: {perc_1:.1f}%"
            
            cv2.putText(annotated_frame, stat_text, (50, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # SIMPAN DATA KE LIST UNTUK CSV (Forward Filling otomatis)
            possession_records.append({
                'frame': frame_idx,
                'team_id': last_possession_team,
                'pink_perc': round(perc_0, 2),
                'blue_perc': round(perc_1, 2)
            })

        cv2.putText(annotated_frame, possession_label, (50, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3)

        # Resolusi Tim Kiper & Wasit
        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(players, players_team_id, goalkeepers)
        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        # Merge semua untuk Anotasi
        detections_all = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
            players_team_id.tolist() +
            goalkeepers_team_id.tolist() +
            [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in detections_all.tracker_id] if detections_all.tracker_id is not None else []

        # Gambar Anotasi ke Frame
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections_all, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(annotated_frame, detections_all, labels, custom_color_lookup=color_lookup)

        # Render & Overlay Radar
        h, w, _ = frame.shape
        radar = render_radar(detections_all, keypoints, color_lookup, last_possession_team)
        radar = sv.resize_image(radar, (w // 4, h // 4))
        radar_h, radar_w, _ = radar.shape
        rect = sv.Rect(x=w // 2 - radar_w // 2, y=h - radar_h, width=radar_w, height=radar_h)
        annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.5, rect=rect)

        frame_idx += 1
        yield annotated_frame

    # 5. Save Statistik Final ke CSV
    if possession_records:
        df_stats = pd.DataFrame(possession_records)
        df_stats.to_csv(output_csv, index=False)
        print(f"\n[INFO] Statistik possession berhasil diekstrak ke {output_csv}")



def main(source_video_path: str, target_video_path: str, device: str, mode: Mode, method: str) -> None:
    if mode == Mode.PITCH_DETECTION:
        frame_generator = run_pitch_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.PLAYER_DETECTION:
        frame_generator = run_player_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.BALL_DETECTION:
        frame_generator = run_ball_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.PLAYER_TRACKING:
        frame_generator = run_player_tracking(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.TEAM_CLASSIFICATION:
        frame_generator = run_team_classification(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.RADAR:
        frame_generator = run_radar(
            source_video_path=source_video_path, device=device, method=method)
    elif mode == Mode.EXTRACT:
        run_tracking_extraction(source_video_path, device, "hasil_ekstraksi.csv")
        return # Keluar dari main karena tidak ada video yang perlu di-sink
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented.")

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in frame_generator:
            sink.write_frame(frame)

        #     cv2.imshow("frame", frame)
        #     if cv2.waitKey(1) & 0xFF == ord("q"):
        #         break
        # cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--source_video_path', type=str, required=True)
    parser.add_argument('--target_video_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--mode', type=Mode, default=Mode.RADAR)
    parser.add_argument('--method', type=str, default='euclidean', 
                        choices=['euclidean', 'manhattan', 'canberra'],
                        help="Metode perhitungan jarak")
    args = parser.parse_args()
    main(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        device=args.device,
        mode=args.mode,
        method=args.method
    )
