from scipy.signal import find_peaks
import pandas as pd
import numpy as np


def get_anomaly_indexes(score, anomaly_score_method, anomaly_score_param):
    scores = np.asarray(score, dtype=np.float64)

    # 0) 비유한값(NaN/Inf) 위치는 무조건 이상치로 채택
    invalid_mask = ~np.isfinite(scores)
    invalid_idx = np.where(invalid_mask)[0]

    if anomaly_score_param is None:
        anomaly_score_param = anomaly_score_default_param()

    if anomaly_score_method == "std":
        idx_core, _ = detect_anomalies_from_scores(scores, anomaly_score_param['threshold'],
                                                   anomaly_score_param['std_factor'])

    elif anomaly_score_method == "percentile":
        idx_core, _ = percentile_thresholding(scores, anomaly_score_param['percentile'])

    elif anomaly_score_method == "peak":
        idx_core, _ = peak_based_anomalies(scores, anomaly_score_param['height'],
                                           anomaly_score_param['distance'])

    elif anomaly_score_method == "adaptive":
        idx_core = adaptive_thresholding(scores, anomaly_score_param['window_size'],
                                         anomaly_score_param['std_factor'])

    else:
        raise ValueError(f"Unknown anomaly detection method: {anomaly_score_method}")

    # 1) 최종: 점수 기반 + (NaN/Inf 위치) 합집합
    if idx_core is None or len(idx_core) == 0:
        return invalid_idx
    return np.unique(np.concatenate([idx_core, invalid_idx]))


def detect_anomalies_from_scores(scores, threshold, std_factor):
    scores = np.asarray(scores, dtype=np.float64)

    # NaN/Inf는 무시하고 통계 계산
    finite = np.isfinite(scores)
    if threshold is None:
        if not finite.any():
            # 전부 비유한값이면 임계값 계산 불가 → 이상치 없음(상위에서 invalid_idx가 처리)
            return np.array([], dtype=int), np.nan
        mean = np.nanmean(scores[finite])
        std = np.nanstd(scores[finite])
        threshold = mean + std_factor * std
        print("========================== ", threshold, " ========================== ")

    # 임계값이 NaN이면 실패 → 이상치 비우기(상위에서 invalid_idx 합쳐짐)
    if not np.isfinite(threshold):
        return np.array([], dtype=int), threshold

    anomaly_indices = np.where(scores >= threshold)[0]
    return anomaly_indices, threshold


def percentile_thresholding(scores, percentile):
    scores = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(scores)
    if not finite.any():
        # 모든 값이 비유한값 → 퍼센타일 불가
        return np.array([], dtype=int), np.nan

    threshold = np.nanpercentile(scores[finite], percentile)  # NaN 무시
    anomaly_indices = np.where(scores >= threshold)[0]        # 동치 포함
    print("========================== ", threshold, " ========================== ")
    return anomaly_indices, threshold


def peak_based_anomalies(scores, height, distance):
    scores = np.asarray(scores, dtype=np.float64)
    # NaN/Inf는 피크 탐지에서 떨어뜨리기 위해 -inf로 치환
    s = scores.copy()
    s[~np.isfinite(s)] = -np.inf
    # height가 None이면 find_peaks는 높이 제한 없이 탐지
    peaks, props = find_peaks(s, height=height, distance=distance)
    # props["peak_heights"]는 존재하되, -inf 치환으로 비유한 위치는 자동 배제됨
    return peaks, props.get("peak_heights", np.array([]))


def adaptive_thresholding(scores, window_size, std_factor):
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return np.array([], dtype=int)

    anomaly_indices = []
    for i in range(window_size, n):
        window = scores[i - window_size:i]
        finite = np.isfinite(window)
        # 윈도우 내 유효값 없으면 임계 계산 불가 → 현재 포인트가 유효하고 커도 판단 불가, 비유한이면 이상으로 플래그
        if not finite.any():
            if not np.isfinite(scores[i]):
                anomaly_indices.append(i)
            continue

        local_mean = np.nanmean(window[finite])
        local_std = np.nanstd(window[finite])
        threshold = local_mean + std_factor * local_std

        # 현재 포인트가 비유한값이면 바로 이상
        if not np.isfinite(scores[i]):
            anomaly_indices.append(i)
        else:
            if np.isfinite(threshold) and scores[i] > threshold:
                anomaly_indices.append(i)

    return np.array(anomaly_indices, dtype=int)


def anomaly_score_default_param():
    return {
        "std": {
            "threshold": None,
            "std_factor": 3.0,
        },
        "percentile": {
            "percentile": 99.5,
        },
        "peak": {
            "height": None,
            "distance": 10,
        },
        "adaptive": {
            "window_size": 100,
            "std_factor": 3.0,
        }
    }
