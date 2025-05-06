# src/utils/analyzer.py
import numpy as np
import math
from scipy.signal import find_peaks
import scipy.stats as stats
from .constants import BODY_PARTS

class ASDBehaviorAnalyzer:
    def __init__(self, keypoints, fps):
        self.keypoints = keypoints
        self.fps = fps
        self.MIN_CONFIDENCE = 0.2

    def _get_part_trajectory(self, part):
        trajectory = []
        for kps in self.keypoints:
            if kps is None or not kps[0][2]:
                trajectory.append([None, None, None])
                continue
            idx = BODY_PARTS[part]
            x, y, conf = kps[idx]
            trajectory.append([x, y, conf] if conf > self.MIN_CONFIDENCE else [None, None, None])
        return trajectory

    def _calculate_velocity(self, positions):
        velocities = []
        for i in range(1, len(positions)):
            if None in positions[i] or None in positions[i-1]:
                velocities.append(0)
                continue
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            dt = 1 / self.fps
            velocity = math.sqrt(dx**2 + dy**2) / dt
            velocities.append(velocity)
        return velocities

    def _calculate_acceleration(self, velocities):
        accelerations = []
        for i in range(1, len(velocities)):
            dv = velocities[i] - velocities[i-1]
            dt = 1 / self.fps
            acceleration = dv / dt
            accelerations.append(acceleration)
        return accelerations

    def _detect_arm_flapping(self, wrist_positions, elbow_positions):
        if len(wrist_positions) < 3:
            return 0, 0, 0, 0
        
        angles = []
        for i in range(len(wrist_positions)):
            if None in wrist_positions[i] or None in elbow_positions[i] or i == 0 or i == len(wrist_positions)-1:
                angles.append(None)
                continue
            vec = (wrist_positions[i][0] - elbow_positions[i][0], wrist_positions[i][1] - elbow_positions[i][1])
            norm = math.sqrt(vec[0]**2 + vec[1]**2)
            if norm > 0:
                angles.append(math.atan2(vec[1], vec[0]))
            else:
                angles.append(None)
        
        angular_velocities = []
        valid_angles = [a for a in angles if a is not None]
        for i in range(1, len(valid_angles)):
            delta = abs(valid_angles[i] - valid_angles[i-1])
            delta = min(delta, 2*math.pi - delta)
            angular_velocities.append(delta * self.fps)
        
        if not angular_velocities:
            return 0, 0, 0, 0
        
        peaks, _ = find_peaks(angular_velocities, height=math.pi/8, distance=2)
        flap_count = len(peaks)
        mean_flap_rate = flap_count / (len(valid_angles)/self.fps) if len(valid_angles) > 0 else 0
        mean_angular_velocity = np.mean(angular_velocities) if len(angular_velocities) > 0 else 0
        
        if len(angular_velocities) > 3:
            fft = np.fft.fft(np.array(angular_velocities) - np.mean(angular_velocities))
            freqs = np.fft.fftfreq(len(angular_velocities), d=1/self.fps)
            dominant_idx = np.argmax(np.abs(fft)[1:]) + 1
            flap_power = np.abs(fft[dominant_idx])
        else:
            flap_power = 0
        
        is_flapping = 1 if (flap_count >= 1 and mean_flap_rate >= 0.5 and mean_angular_velocity >= math.pi/8) else 0
        return is_flapping, flap_count, mean_flap_rate, flap_power

    def _detect_head_banging(self):
        head_positions = self._get_part_trajectory("Nose")
        if len(head_positions) < 3:
            return 0, 0, 0, 0
        
        y_positions = [p[1] for p in head_positions if p[1] is not None]
        if len(y_positions) < 3:
            return 0, 0, 0, 0
        
        velocities = self._calculate_velocity([(0, y) for y in y_positions])
        accelerations = self._calculate_acceleration(velocities)
        
        bang_threshold = -20
        bang_indices = [i for i, a in enumerate(accelerations) if a < bang_threshold]
        
        bang_count = len(bang_indices)
        bang_rate = bang_count / (len(y_positions)/self.fps) if len(y_positions) > 0 else 0
        mean_accel = np.mean(accelerations) if len(accelerations) > 0 else 0
        accel_variance = np.var(accelerations) if len(accelerations) > 1 else 0
        
        is_banging = 1 if (bang_count >= 1 and bang_rate >= 0.2 and mean_accel < -5) else 0
        return is_banging, bang_count, bang_rate, accel_variance

    def _detect_body_rocking(self):
        hip_positions = self._get_part_trajectory("MidHip")
        if len(hip_positions) < 3:
            return 0, 0, 0, 0
        
        x_positions = [p[0] for p in hip_positions if p[0] is not None]
        if len(x_positions) < 3:
            return 0, 0, 0, 0
        
        x_positions = np.array(x_positions)
        fft = np.fft.fft(x_positions - np.mean(x_positions))
        freqs = np.fft.fftfreq(len(x_positions), d=1/self.fps)
        
        dominant_idx = np.argmax(np.abs(fft)[1:]) + 1
        dominant_freq = abs(freqs[dominant_idx])
        dominant_power = np.abs(fft[dominant_idx])
        
        x_diff = np.diff(x_positions)
        x_diff = x_diff[x_diff != 0]
        entropy = stats.entropy(np.histogram(x_diff, bins=20, density=True)[0]) if len(x_diff) > 1 else 0
        
        is_rocking = 1 if (0.2 <= dominant_freq <= 3.0 and dominant_power > 20) else 0
        return is_rocking, dominant_freq, dominant_power, entropy

    def _detect_spinning(self):
        if len(self.keypoints) < 3:
            return 0, 0, 0, 0
        
        angles = []
        for kps in self.keypoints:
            if kps is None or not kps[0][2]:
                angles.append(None)
                continue
            rshoulder = kps[BODY_PARTS["RShoulder"]]
            lshoulder = kps[BODY_PARTS["LShoulder"]]
            rhip = kps[BODY_PARTS["RHip"]]
            lhip = kps[BODY_PARTS["LHip"]]
            if None in rshoulder + lshoulder + rhip + lhip:
                angles.append(None)
                continue
            shoulder_vec = (rshoulder[0] - lshoulder[0], rshoulder[1] - lshoulder[1])
            hip_vec = (rhip[0] - lhip[0], rhip[1] - lhip[1])
            dot = shoulder_vec[0]*hip_vec[0] + shoulder_vec[1]*hip_vec[1]
            det = shoulder_vec[0]*hip_vec[1] - shoulder_vec[1]*hip_vec[0]
            angle = math.atan2(det, dot)
            angles.append(angle)
        
        valid_angles = [a for a in angles if a is not None]
        angular_velocities = []
        for i in range(1, len(valid_angles)):
            delta = valid_angles[i] - valid_angles[i-1]
            delta = (delta + math.pi) % (2*math.pi) - math.pi
            angular_velocities.append(abs(delta) * self.fps)
        
        if not angular_velocities:
            return 0, 0, 0, 0
        
        spin_threshold = math.pi/8
        spin_duration = 0.3
        min_spin_frames = int(spin_duration * self.fps)
        
        spin_frames = sum(1 for v in angular_velocities if v > spin_threshold)
        mean_spin_velocity = np.mean(angular_velocities) if len(angular_velocities) > 0 else 0
        angle_variance = np.var(valid_angles) if len(valid_angles) > 1 else 0
        is_spinning = 1 if spin_frames >= min_spin_frames else 0
        return is_spinning, spin_frames/len(angular_velocities), mean_spin_velocity, angle_variance

    def _calculate_elbow_angle(self, shoulder, elbow, wrist):
        if None in shoulder + elbow + wrist:
            return None
        v1 = (shoulder[0] - elbow[0], shoulder[1] - elbow[1])
        v2 = (wrist[0] - elbow[0], wrist[1] - elbow[1])
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        det = v1[0]*v2[1] - v1[1]*v2[0]
        angle = math.atan2(det, dot)
        return abs(angle)

    def extract_features(self):
        features = {}
        
        # Arm flapping
        left_wrist_positions = self._get_part_trajectory("LWrist")
        right_wrist_positions = self._get_part_trajectory("RWrist")
        left_elbow_positions = self._get_part_trajectory("LElbow")
        right_elbow_positions = self._get_part_trajectory("RElbow")
        
        left_flap, left_flap_count, left_flap_rate, left_flap_power = self._detect_arm_flapping(
            left_wrist_positions, left_elbow_positions)
        right_flap, right_flap_count, right_flap_rate, right_flap_power = self._detect_arm_flapping(
            right_wrist_positions, right_elbow_positions)
        
        features["arm_flapping"] = max(left_flap, right_flap)
        features["left_flap_count"] = left_flap_count
        features["right_flap_count"] = right_flap_count
        features["mean_flap_rate"] = (left_flap_rate + right_flap_rate) / 2 if left_flap_rate + right_flap_rate > 0 else 0
        features["mean_flap_power"] = (left_flap_power + right_flap_power) / 2 if left_flap_power + right_flap_power > 0 else 0
        
        # Head banging
        head_banging, head_bang_count, head_bang_rate, head_accel_variance = self._detect_head_banging()
        features["head_banging"] = head_banging
        features["head_bang_count"] = head_bang_count
        features["head_bang_rate"] = head_bang_rate
        features["head_accel_variance"] = head_accel_variance
        
        # Body rocking
        body_rocking, rocking_frequency, rocking_power, rocking_entropy = self._detect_body_rocking()
        features["body_rocking"] = body_rocking
        features["rocking_frequency"] = rocking_frequency
        features["rocking_power"] = rocking_power
        features["rocking_entropy"] = rocking_entropy
        
        # Spinning
        spinning, spin_ratio, spin_velocity, spin_angle_variance = self._detect_spinning()
        features["spinning"] = spinning
        features["spin_ratio"] = spin_ratio
        features["spin_velocity"] = spin_velocity
        features["spin_angle_variance"] = spin_angle_variance
        
        # Additional kinematic features
        left_wrist_velocities = self._calculate_velocity([p for p in left_wrist_positions if p[0] is not None])
        right_wrist_velocities = self._calculate_velocity([p for p in right_wrist_positions if p[0] is not None])
        nose_positions = self._get_part_trajectory("Nose")
        nose_velocities = self._calculate_velocity([p for p in nose_positions if p[0] is not None])
        nose_accelerations = self._calculate_acceleration(nose_velocities)
        
        features["LWrist_mean_velocity"] = np.mean(left_wrist_velocities) if left_wrist_velocities else 0
        features["LWrist_max_velocity"] = np.max(left_wrist_velocities) if left_wrist_velocities else 0
        features["LWrist_movement_variability"] = np.var(left_wrist_velocities) if left_wrist_velocities else 0
        features["RWrist_mean_velocity"] = np.mean(right_wrist_velocities) if right_wrist_velocities else 0
        features["RWrist_skewness"] = stats.skew(right_wrist_velocities) if len(right_wrist_velocities) > 2 else 0
        features["Nose_mean_acceleration"] = np.mean(nose_accelerations) if nose_accelerations else 0
        features["wrist_velocity_diff"] = abs(features["LWrist_mean_velocity"] - features["RWrist_mean_velocity"])
        
        # Wrist-nose distance and variability
        wrist_nose_distances = []
        for lw, rw, nose in zip(left_wrist_positions, right_wrist_positions, nose_positions):
            if None in lw + rw + nose:
                continue
            dist_l = math.sqrt((lw[0] - nose[0])**2 + (lw[1] - nose[1])**2)
            dist_r = math.sqrt((rw[0] - nose[0])**2 + (rw[1] - nose[1])**2)
            wrist_nose_distances.append((dist_l + dist_r) / 2)
        
        features["mean_wrist_nose_distance"] = np.mean(wrist_nose_distances) if wrist_nose_distances else 0
        features["wrist_nose_variability"] = np.var(wrist_nose_distances) if wrist_nose_distances else 0
        left_wrist_variances = [np.var([p[0], p[1]]) for p in left_wrist_positions if p[0] is not None]
        right_wrist_variances = [np.var([p[0], p[1]]) for p in right_wrist_positions if p[0] is not None]
        features["wrist_variability_diff"] = (np.mean(left_wrist_variances) - np.mean(right_wrist_variances)) if left_wrist_variances and right_wrist_variances else 0
        
        # Elbow angle variability
        left_elbow_angles = []
        right_elbow_angles = []
        for kps in self.keypoints:
            if kps is None or not kps[0][2]:
                left_elbow_angles.append(None)
                right_elbow_angles.append(None)
                continue
            left_angle = self._calculate_elbow_angle(
                kps[BODY_PARTS["LShoulder"]], kps[BODY_PARTS["LElbow"]], kps[BODY_PARTS["LWrist"]])
            right_angle = self._calculate_elbow_angle(
                kps[BODY_PARTS["RShoulder"]], kps[BODY_PARTS["RElbow"]], kps[BODY_PARTS["RWrist"]])
            left_elbow_angles.append(left_angle)
            right_elbow_angles.append(right_angle)
        
        valid_left_angles = [a for a in left_elbow_angles if a is not None]
        valid_right_angles = [a for a in right_elbow_angles if a is not None]
        features["elbow_angle_variability"] = (np.var(valid_left_angles) + np.var(valid_right_angles)) / 2 if valid_left_angles and valid_right_angles else 0
        
        # Video metadata
        features["duration"] = len(self.keypoints) / self.fps if self.fps > 0 else 0
        features["frame_count"] = len(self.keypoints)
        
        return features