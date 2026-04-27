from scipy.spatial import ConvexHull
import torch
import yaml
from modules.keypoint_detector import KPDetector
from modules.generator_optim import OcclusionAwareGenerator
from sync_batchnorm import DataParallelWithCallback
import numpy as np
import face_alignment
import cv2
import os
import sys

# Add face-animation to path for GPEN access
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'face-animation'))



def regularize_jacobian(jacobian, stabilization=1.0):
    """Force the Jacobian matrix to be isotropic (no shearing).
    Allows for uniform scaling and rotation, but eliminates warping.
    """
    if stabilization <= 0:
        return jacobian
        
    U, S, V = torch.svd(jacobian)
    
    # Isotropic scaling: make both singular values equal to their average
    # This preserves the area/scale but eliminates shearing
    S_mean = S.mean(dim=-1, keepdim=True).repeat(1, 1, 2)
    S_new = (1.0 - stabilization) * S + stabilization * S_mean
    
    # Reconstruct
    R = torch.matmul(U, torch.matmul(torch.diag_embed(S_new), V.transpose(-1, -2)))
    
    # Correct for potential reflections (ensure positive determinant)
    det = torch.det(R)
    V_fixed = V.clone()
    V_fixed[..., 1] *= torch.sign(det).unsqueeze(-1)
    R = torch.matmul(U, torch.matmul(torch.diag_embed(S_new), V_fixed.transpose(-1, -2)))
    return R


def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False, **kwargs):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
        scale = np.sqrt(source_area) / np.sqrt(driving_area)
        scale = np.clip(scale, 0.95, 1.05) 
    else:
        scale = 1.0

    kp_new = {k: v.clone() for k, v in kp_driving.items()}
    mouth_indices = kwargs.get('mouth_kp_indices', [])
    eye_indices = kwargs.get('eye_kp_indices', [])
    eyebrow_indices = kwargs.get('eyebrow_kp_indices', [])
    jaw_indices = kwargs.get('jaw_kp_indices', [])
    nose_indices = kwargs.get('nose_kp_indices', [])
    neck_indices = kwargs.get('neck_kp_indices', [])
    
    face_indices = (mouth_indices if mouth_indices else []) + \
                   (eye_indices if eye_indices else []) + \
                   (eyebrow_indices if eyebrow_indices else []) + \
                   (jaw_indices if jaw_indices else []) + \
                   (nose_indices if nose_indices else []) + \
                   (neck_indices if neck_indices else [])
    
    # 1. SURGICAL SEMANTIC TRACKING: Identify Background
    all_indices = set(range(kp_driving['value'].shape[1]))
    background_indices = list(all_indices - set(face_indices))

    if use_relative_movement:
        # BLOCK-MOVEMENT STABILIZATION
        if face_indices:
            driving_centroid = kp_driving['value'][:, face_indices].mean(dim=1, keepdim=True)
            initial_centroid = kp_driving_initial['value'][:, face_indices].mean(dim=1, keepdim=True)
            global_delta = (driving_centroid - initial_centroid)
            local_delta = (kp_driving['value'] - driving_centroid) - (kp_driving_initial['value'] - initial_centroid)
            
            # REHAUL: Calculate distance-based falloff for global movement
            # This ensures only the head moves, while outer areas stay anchored to prevent stretching.
            dist_to_centroid = torch.norm(kp_driving['value'] - driving_centroid, dim=2, keepdim=True)
            # Use mouth/eyes spread to define face radius
            face_spread = torch.std(kp_driving['value'][:, face_indices], dim=1, keepdim=True).mean(dim=-1, keepdim=True)
            face_radius = face_spread * 2.5
            
            # Radial falloff: 1.0 at center, fades to 0.0 at 2x radius
            radial_falloff = torch.clamp(1.0 - (dist_to_centroid - face_radius) / (face_radius * 1.5), 0.0, 1.0)
            
            turn_intensity = torch.abs(global_delta[:, :, 0:1]) * 1.0
            turn_factor = torch.clamp(torch.exp(-turn_intensity), 0.8, 1.0)
            
            # 2. SURGICAL JAW PROTECTION: Jaw/Mouth are 100% responsive
            bg_mask = torch.zeros(kp_driving['value'].shape[1], device=kp_driving['value'].device).view(1, -1, 1)
            bg_mask[0, background_indices] = 1.0
            
            expression_mask = torch.zeros(kp_driving['value'].shape[1], device=kp_driving['value'].device).view(1, -1, 1)
            expression_mask[0, mouth_indices + jaw_indices] = 1.0
            
            nose_mask = torch.zeros(kp_driving['value'].shape[1], device=kp_driving['value'].device).view(1, -1, 1)
            if nose_indices:
                nose_mask[0, nose_indices] = 1.0
            
            neck_mask = torch.zeros(kp_driving['value'].shape[1], device=kp_driving['value'].device).view(1, -1, 1)
            if neck_indices:
                neck_mask[0, neck_indices] = 1.0
            
            # Mouth Sensitivity: Boost mouth movement for smaller characters
            sensitivity = kwargs.get('mouth_sensitivity', 1.0)
            mouth_mask = torch.zeros(kp_driving['value'].shape[1], device=kp_driving['value'].device).view(1, -1, 1)
            if mouth_indices:
                mouth_mask[0, mouth_indices] = 1.0
            
            # Move Factor: How much local keypoints move relative to head centroid
            move_factor = (1.0 - bg_mask) * (neck_mask * 0.6 + nose_mask * 1.0 + (1.0 - nose_mask - neck_mask) * (expression_mask * 1.0 + (1.0 - expression_mask) * turn_factor)) + (bg_mask * 0.0)
            
            # Apply Mouth Sensitivity to local expressions
            local_delta = local_delta * (1.0 + mouth_mask * (sensitivity - 1.0))
            
            # REHAUL: Global movement intensity follows the radial falloff
            # Points far from the head (shoulders, hair ends) won't move, stopping the "stretch"
            global_intensity = radial_falloff * (1.0 - bg_mask)
            kp_value_diff = (global_delta * global_intensity) + (local_delta * move_factor)
        else:
            kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
            kp_value_diff[:, :, 0] *= 0.85
            
        kp_new['value'] = kp_value_diff * scale + kp_source['value']
    else:
        kp_new['value'] = kp_driving['value']

    if use_relative_jacobian:
        jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
        
        # 2. SEMANTIC JACOBIAN LOCKING
        if face_indices:
            # Shared face rotation
            head_jac = jacobian_diff[:, face_indices].mean(dim=1, keepdim=True)
            U, S, V = torch.svd(head_jac)
            # Force Pure Rotation for the head (Isotropic Head)
            head_jac = torch.matmul(U, torch.matmul(torch.diag_embed(torch.ones_like(S)), V.transpose(-1, -2)))
            
            # Background Rotation Locked to Zero (Identity)
            bg_mask = torch.zeros(jacobian_diff.shape[1], device=jacobian_diff.device).view(1, -1, 1, 1)
            bg_mask[0, background_indices] = 1.0
            
            # Neck should also be relatively rigid
            neck_mask = torch.zeros(jacobian_diff.shape[1], device=jacobian_diff.device).view(1, -1, 1, 1)
            if neck_indices:
                neck_mask[0, neck_indices] = 1.0
                
            # Blend: face has tilt, neck has half tilt, hair/bg is perfectly static
            jacobian_diff = (1.0 - bg_mask - neck_mask) * jacobian_diff + (neck_mask * 0.5 * (jacobian_diff + torch.eye(2, device=jacobian_diff.device).view(1, 1, 2, 2))) + bg_mask * torch.eye(2, device=jacobian_diff.device).view(1, 1, 2, 2)

        # Apply stabilization/dampening
        stabilization = kwargs.get('jacobian_stabilization', 0.5)
        # Aggressive Isotropic Fix: High dampening forces near-perfect uniformity
        dampening = kwargs.get('jacobian_dampening', 0.95)
        
        jacobian_diff = regularize_jacobian(jacobian_diff, stabilization=stabilization)

        # Final Isotropic fix with STRICT CLAMPING
        U, S, V = torch.svd(jacobian_diff)
        S_mean = S.mean(dim=-1, keepdim=True).repeat(1, 1, 2)
        S_dampened = (1.0 - dampening) * S + dampening * S_mean
        
        # Scaling Limiters: Prevent extreme stretching (>1.03x or <0.97x) to stop face from deforming when turning
        S_dampened = torch.clamp(S_dampened, 0.97, 1.03)
        
        jacobian_diff = torch.matmul(U, torch.matmul(torch.diag_embed(S_dampened), V.transpose(-1, -2)))

        kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])
    else:
        kp_new['jacobian'] = kp_driving['jacobian']

    return kp_new


def to_tensor(a):
    return torch.tensor(a[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2) / 255


class PredictorLocal:
    def __init__(self, config_path, checkpoint_path, relative=False, adapt_movement_scale=False, device=None, enc_downscale=1, smooth_factor=0.1, no_relative_jacobian=False, jacobian_stabilization=0.4, **kwargs):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.relative = relative
        self.adapt_movement_scale = adapt_movement_scale
        self.smooth_factor = smooth_factor
        self.no_relative_jacobian = no_relative_jacobian
        self.jacobian_stabilization = jacobian_stabilization
        self.jacobian_dampening = kwargs.get('jacobian_dampening', 0.9)
        self.mouth_sensitivity = 1.0
        self.kp_driving_smoothed = None
        self.start_frame = None
        self.start_frame_kp = None
        self.kp_driving_initial = None
        self.mouth_kp_indices = None
        self.eye_kp_indices = None
        self.eyebrow_kp_indices = None
        self.jaw_kp_indices = None
        self.nose_kp_indices = None
        self.cheek_kp_indices = None
        self.background_indices = None
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.generator, self.kp_detector = self.load_checkpoints()
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True, device=self.device)
        self.source = None
        self.kp_source = None
        self.enc_downscale = enc_downscale
        
        self.enhance = kwargs.get('enhance', False)
        if self.enhance:
            from GPEN.face_enhancement import FaceEnhancement
            self.face_enhancer = FaceEnhancement(
                size=256, model="GPEN-BFR-256", use_sr=False, sr_model="realesrnet_x2", channel_multiplier=1, narrow=0.5, use_facegan=True
            )
    
    def load_checkpoints(self):
        with open(self.config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    
        generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                            **config['model_params']['common_params'])
        generator.to(self.device)
    
        kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                                 **config['model_params']['common_params'])
        kp_detector.to(self.device)
    
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        generator.load_state_dict(checkpoint['generator'])
        kp_detector.load_state_dict(checkpoint['kp_detector'])
    
        generator.eval()
        kp_detector.eval()
        
        return generator, kp_detector

    def reset_frames(self):
        self.kp_driving_initial = None
        self.kp_driving_smoothed = None
        self.kp_driving_prev = None
        self.mouth_kp_indices = None
        self.eye_kp_indices = None
        self.eyebrow_kp_indices = None
        self.jaw_kp_indices = None
        self.nose_kp_indices = None
        self.cheek_kp_indices = None
        self.background_indices = None
        self.frame_count = 0

    def set_source_image(self, source_image):
        self.source = to_tensor(source_image).to(self.device)
        self.kp_source = self.kp_detector(self.source)
        
        lmk = self.get_frame_landmarks(source_image)
        if lmk is not None:
            # Calculate Mouth Sensitivity
            inner_mouth = lmk[60:68]
            eye_dist = np.linalg.norm(lmk[36] - lmk[45])
            mouth_size = np.linalg.norm(inner_mouth[0] - inner_mouth[4])
            self.mouth_sensitivity = np.clip(0.4 / (mouth_size / (eye_dist + 1e-6)), 0.8, 1.5)
            print(f"Mouth sensitivity boosted to {self.mouth_sensitivity:.2f}")

        if self.enc_downscale > 1:
            h, w = int(self.source.shape[2] / self.enc_downscale), int(self.source.shape[3] / self.enc_downscale)
            source_enc = torch.nn.functional.interpolate(self.source, size=(h, w), mode='bilinear')
        else:
            source_enc = self.source

        self.generator.encode_source(source_enc)

    def identify_semantic_kp(self, kp_driving, face_landmarks):
        """Map unsupervised FOMM keypoints to facial features using 68-pt landmarks"""
        mouth_indices_fa = list(range(48, 68))
        eye_indices_fa = list(range(36, 48))
        brow_indices_fa = list(range(17, 27))
        jaw_indices_fa = list(range(0, 17))
        nose_indices_fa = list(range(27, 36))
        
        mouth_landmarks = face_landmarks[mouth_indices_fa]
        eye_landmarks = face_landmarks[eye_indices_fa]
        brow_landmarks = face_landmarks[brow_indices_fa]
        jaw_landmarks = face_landmarks[jaw_indices_fa]
        nose_landmarks = face_landmarks[nose_indices_fa]

        # 0. NECK/CHIN ESTIMATION: Landmarks 4, 8, 12 are bottom of jaw
        neck_landmarks = face_landmarks[[4, 8, 12]]
        # Shift them down to estimate neck area
        neck_landmarks[:, 1] += 20 

        # Normalize landmarks to -1, 1 range to match FOMM keypoints
        h, w = 256, 256
        mouth_landmarks = (mouth_landmarks / (h / 2)) - 1.0
        eye_landmarks = (eye_landmarks / (h / 2)) - 1.0
        brow_landmarks = (brow_landmarks / (h / 2)) - 1.0
        jaw_landmarks = (jaw_landmarks / (h / 2)) - 1.0
        nose_landmarks = (nose_landmarks / (h / 2)) - 1.0
        neck_landmarks = (neck_landmarks / (h / 2)) - 1.0
        face_landmarks_norm = (face_landmarks / (h / 2)) - 1.0

        kp_values = kp_driving['value'][0].cpu().numpy()

        mouth_indices = []
        eye_indices = []
        brow_indices = []
        jaw_indices = []
        nose_indices = []
        cheek_indices = []
        neck_indices = []
        background_indices = []

        for i, kp in enumerate(kp_values):
            dist_to_face = np.min(np.linalg.norm(face_landmarks_norm - kp, axis=1))
            # Mouth
            if np.min(np.linalg.norm(mouth_landmarks - kp, axis=1)) < 0.25:
                mouth_indices.append(i)
            # Eyes
            elif np.min(np.linalg.norm(eye_landmarks - kp, axis=1)) < 0.20:
                eye_indices.append(i)
            # Brows
            elif np.min(np.linalg.norm(brow_landmarks - kp, axis=1)) < 0.25:
                brow_indices.append(i)
            # Nose
            elif np.min(np.linalg.norm(nose_landmarks - kp, axis=1)) < 0.20:
                nose_indices.append(i)
            # Jawline
            elif np.min(np.linalg.norm(jaw_landmarks - kp, axis=1)) < 0.25:
                jaw_indices.append(i)
            # Neck (Points below jaw)
            elif np.min(np.linalg.norm(neck_landmarks - kp, axis=1)) < 0.30:
                neck_indices.append(i)
            # Cheek Check: Points between nose and jaw but not mouth/eyes
            elif dist_to_face < 0.35:
                cheek_indices.append(i)

            # Background Check (Hair/Clothes)
            if dist_to_face > 0.40:
                background_indices.append(i)

        return mouth_indices, eye_indices, brow_indices, jaw_indices, nose_indices, cheek_indices, neck_indices, background_indices
    def get_frame_landmarks(self, frame):
        """Get 68-pt landmarks for a single frame"""
        landmarks = self.fa.get_landmarks(frame)
        if landmarks:
            return landmarks[0]
        return None

    def get_kp_indices(self, kp_values, landmarks):
        """Map keypoints to closest landmarks"""
        kp_np = kp_values[0].cpu().numpy()
        h, w = 256, 256
        landmarks_norm = (landmarks / (h / 2)) - 1.0
        
        indices = []
        for i, kp in enumerate(kp_np):
            if np.min(np.linalg.norm(landmarks_norm - kp, axis=1)) < 0.25:
                indices.append(i)
        return indices

    def get_gpen_landmarks(self, fa_landmarks):
        """Extract 5-pt landmarks for GPEN alignment from 68-pt landmarks"""
        # [left_eye, right_eye, nose, left_mouth, right_mouth]
        left_eye = fa_landmarks[36:42].mean(axis=0)
        right_eye = fa_landmarks[42:48].mean(axis=0)
        nose = fa_landmarks[30]
        left_mouth = fa_landmarks[48]
        right_mouth = fa_landmarks[54]
        return np.array([left_eye, right_eye, nose, left_mouth, right_mouth], dtype=np.float32)

    def predict(self, driving_frame):
        assert self.kp_source is not None, "call set_source_image()"
        self.frame_count = getattr(self, 'frame_count', 0) + 1
        
        with torch.no_grad():
            driving = to_tensor(driving_frame).to(self.device)

            if self.kp_driving_initial is None:
                self.kp_driving_initial = self.kp_detector(driving)
                self.kp_driving_smoothed = None
                self.start_frame = driving_frame.copy()
                self.mouth_kp_indices = None
                self.eye_kp_indices = None
                self.eyebrow_kp_indices = None
                self.jaw_kp_indices = None
                self.nose_kp_indices = None
                self.cheek_kp_indices = None
                self.background_indices = None

            kp_driving = self.kp_detector(driving)

            if self.kp_driving_smoothed is None:
                self.kp_driving_smoothed = {k: v.clone() for k, v in kp_driving.items()}
            else:
                for k in kp_driving.keys():
                    # Calculate movement speed
                    diff = torch.abs(kp_driving[k] - self.kp_driving_smoothed[k]).mean()
                    
                    # Adaptive alpha: move faster -> higher alpha -> more responsive
                    # Base alpha 0.1, goes up to 0.8 for fast movements
                    alpha = self.smooth_factor + (0.95 - self.smooth_factor) * torch.clamp(diff / 0.015, 0.0, 1.0)
                    
                    # Correct EMA formula: alpha * new + (1-alpha) * old
                    self.kp_driving_smoothed[k] = alpha * kp_driving[k] + (1.0 - alpha) * self.kp_driving_smoothed[k]
                    
            # Apply the smoothed keypoints
            kp_driving = {k: v.clone() for k, v in self.kp_driving_smoothed.items()}

            use_rel_jac = self.relative and not getattr(self, 'no_relative_jacobian', False)
            
            # Identify semantic keypoints every 100 frames to handle semantic drift
            if self.mouth_kp_indices is None or (getattr(self, 'frame_count', 0) % 100 == 0):
                fa_landmarks = self.get_frame_landmarks(driving_frame)
                if fa_landmarks is not None:
                    self.mouth_kp_indices, self.eye_kp_indices, self.eyebrow_kp_indices, self.jaw_kp_indices, self.nose_kp_indices, self.cheek_kp_indices, self.neck_kp_indices, self.background_indices = \
                        self.identify_semantic_kp(kp_driving, fa_landmarks)

            kp_norm = normalize_kp(self.kp_source, kp_driving, self.kp_driving_initial, 
                                   adapt_movement_scale=self.adapt_movement_scale,
                                   use_relative_movement=self.relative,
                                   use_relative_jacobian=self.no_relative_jacobian == False,
                                   jacobian_stabilization=self.jacobian_stabilization,
                                   jacobian_dampening=self.jacobian_dampening,
                                   mouth_kp_indices=self.mouth_kp_indices,
                                   eye_kp_indices=self.eye_kp_indices,
                                   eyebrow_kp_indices=self.eyebrow_kp_indices,
                                   jaw_kp_indices=self.jaw_kp_indices,
                                   nose_kp_indices=self.nose_kp_indices,
                                   cheek_kp_indices=self.cheek_kp_indices,
                                   neck_kp_indices=self.neck_kp_indices,
                                   mouth_sensitivity=self.mouth_sensitivity)

            out = self.generator(self.source, kp_source=self.kp_source, kp_driving=kp_norm)

            out = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
            out = (np.clip(out, 0, 1) * 255).astype(np.uint8)

            if self.enhance:
                import time
                e_start = time.time()
                
                # Check if enhancement was too slow last time
                if getattr(self, '_skip_enhance', False):
                    self._skip_enhance = False
                    if self.verbose:
                        print("GPEN: Skipping frame to maintain sync")
                else:
                    try:
                        out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                        
                        # Optimization: Generated face position is mostly stable.
                        # We find the face in the generated image occasionally and reuse the landmarks.
                        if getattr(self, '_gpen_counter', 0) % 15 == 0 or getattr(self, 'source_lmk_guide', None) is None:
                            # Run landmark detection on the generated image (slow, so we do it rarely)
                            lmks = self.fa.get_landmarks(out_bgr)
                            if lmks:
                                self.source_lmk_guide = [self.get_gpen_landmarks(lmks[0])]
                            elif getattr(self, 'source_lmk_guide', None) is None:
                                # Absolute fallback: center face
                                self.source_lmk_guide = [np.array([[96, 112], [160, 112], [128, 140], [100, 175], [156, 175]], dtype=np.float32)]
                        
                        self._gpen_counter = getattr(self, '_gpen_counter', 0) + 1

                        # Enhance using the guide landmarks
                        enhanced, _, _ = self.face_enhancer.process(out_bgr, landmarks=self.source_lmk_guide)
                        out = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
                        
                        e_time = time.time() - e_start
                        if self.verbose:
                            print(f"GPEN: Enhanced in {e_time*1000:.1f}ms")
                            
                        # If enhancement is too slow (>150ms), skip next frame to catch up
                        if e_time > 0.150:
                            self._skip_enhance = True
                    except Exception as e:
                        print(f"GPEN Error: {e}")
                        # Fallback to original image if enhancement fails

            return out


    def get_frame_landmarks(self, image):
        # image is RGB
        # face_alignment expects BGR or RGB? Usually BGR if using cv2, but predictor uses RGB
        # fa.get_landmarks expects (H, W, 3)
        landmarks = self.fa.get_landmarks(image)
        if landmarks:
            return landmarks[0]
        return None

    def get_frame_kp(self, image):
        kp_landmarks = self.fa.get_landmarks(image)
        if kp_landmarks:
            kp_image = kp_landmarks[0]
            kp_image = self.normalize_alignment_kp(kp_image)
            return kp_image
        else:
            return None

    @staticmethod
    def normalize_alignment_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp
    
    def get_start_frame(self):
        return self.start_frame

    def get_start_frame_kp(self):
        return self.start_frame_kp
