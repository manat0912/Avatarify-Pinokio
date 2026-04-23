import os
import sys
import cv2
import yaml
import numpy as np
import torch
import torch.nn.functional as F

# Add face-animation/face-vid2vid to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'face-animation', 'face-vid2vid'))

from sync_batchnorm import DataParallelWithCallback
from modules.generator import OcclusionAwareSPADEGenerator
from modules.keypoint_detector import KPDetector, HEEstimator
from animate import normalize_kp as normalize_kp_fvv
import face_alignment
from scipy.spatial import ConvexHull

def to_tensor(a):
    return torch.tensor(a[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2) / 255

def headpose_pred_to_degree(pred):
    device = pred.device
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)
    pred = F.softmax(pred, dim=1)
    degree = torch.sum(pred * idx_tensor, axis=1) * 3 - 99
    return degree

def get_rotation_matrix(yaw, pitch, roll):
    yaw = yaw / 180 * 3.14
    pitch = pitch / 180 * 3.14
    roll = roll / 180 * 3.14

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    pitch_mat = torch.cat([
        torch.ones_like(pitch), torch.zeros_like(pitch), torch.zeros_like(pitch),
        torch.zeros_like(pitch), torch.cos(pitch), -torch.sin(pitch),
        torch.zeros_like(pitch), torch.sin(pitch), torch.cos(pitch),
    ], dim=1).view(-1, 3, 3)

    yaw_mat = torch.cat([
        torch.cos(yaw), torch.zeros_like(yaw), torch.sin(yaw),
        torch.zeros_like(yaw), torch.ones_like(yaw), torch.zeros_like(yaw),
        -torch.sin(yaw), torch.zeros_like(yaw), torch.cos(yaw),
    ], dim=1).view(-1, 3, 3)

    roll_mat = torch.cat([
        torch.cos(roll), -torch.sin(roll), torch.zeros_like(roll),
        torch.sin(roll), torch.cos(roll), torch.zeros_like(roll),
        torch.zeros_like(roll), torch.zeros_like(roll), torch.ones_like(roll),
    ], dim=1).view(-1, 3, 3)

    rot_mat = torch.einsum("bij,bjk,bkm->bim", pitch_mat, yaw_mat, roll_mat)
    return rot_mat

def keypoint_transformation(kp_canonical, he, estimate_jacobian=False, free_view=False, yaw=0, pitch=0, roll=0, output_coord=False):
    kp = kp_canonical["value"]
    if not free_view:
        yaw, pitch, roll = he["yaw"], he["pitch"], he["roll"]
        yaw = headpose_pred_to_degree(yaw)
        pitch = headpose_pred_to_degree(pitch)
        roll = headpose_pred_to_degree(roll)
    else:
        if yaw is not None:
            yaw = torch.tensor([yaw]).cuda() if torch.cuda.is_available() else torch.tensor([yaw])
        else:
            yaw = headpose_pred_to_degree(he["yaw"])
        if pitch is not None:
            pitch = torch.tensor([pitch]).cuda() if torch.cuda.is_available() else torch.tensor([pitch])
        else:
            pitch = headpose_pred_to_degree(he["pitch"])
        if roll is not None:
            roll = torch.tensor([roll]).cuda() if torch.cuda.is_available() else torch.tensor([roll])
        else:
            roll = headpose_pred_to_degree(he["roll"])

    t, exp = he["t"], he["exp"]
    rot_mat = get_rotation_matrix(yaw, pitch, roll)

    kp_rotated = torch.einsum("bmp,bkp->bkm", rot_mat, kp)
    t = t.unsqueeze(1).repeat(1, kp.shape[1], 1)
    kp_t = kp_rotated + t
    exp = exp.view(exp.shape[0], -1, 3)
    kp_transformed = kp_t + exp

    jacobian_transformed = None
    if estimate_jacobian:
        jacobian = kp_canonical["jacobian"]
        jacobian_transformed = torch.einsum("bmp,bkps->bkms", rot_mat, jacobian)

    if output_coord:
        return {"value": kp_transformed, "jacobian": jacobian_transformed}, {
            "yaw": float(yaw.cpu().numpy()),
            "pitch": float(pitch.cpu().numpy()),
            "roll": float(roll.cpu().numpy()),
        }

    return {"value": kp_transformed, "jacobian": jacobian_transformed}

class PredictorFaceAnimation:
    def __init__(self, config_path=None, checkpoint_path=None, relative=True, adapt_movement_scale=True, device=None, smooth_factor=0.75, **kwargs):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.relative = relative
        self.adapt_movement_scale = adapt_movement_scale
        self.smooth_factor = smooth_factor
        
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "..", "face-animation", "face-vid2vid", "config", "vox-256-spade.yaml")
        
        if checkpoint_path is None or checkpoint_path == 'vox-cpk.pth.tar':
            # Use the default FaceMapping checkpoint
            checkpoint_path = os.path.join(os.path.dirname(__file__), "..", "FaceMapping.pth.tar")
            if not os.path.exists(checkpoint_path):
                from gdown import download
                file_id = "11ZgyjKI5OcB7klcsIdPpCCX38AIX8Soc"
                print(f"Downloading checkpoint to {checkpoint_path}")
                download(id=file_id, output=checkpoint_path, quiet=False)

        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        
        self.generator, self.kp_detector, self.he_estimator = self.load_checkpoints()
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True, device=self.device)
        
        self.source = None
        self.kp_canonical = None
        self.he_source = None
        self.kp_source = None
        self.kp_driving_initial = None
        self.start_frame = None
        self.start_frame_kp = None
        self.coordinates = None

    def load_checkpoints(self):
        with open(self.config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        generator = OcclusionAwareSPADEGenerator(**config["model_params"]["generator_params"], **config["model_params"]["common_params"])
        if self.device == 'cuda':
            generator.cuda().half()
        else:
            generator.to(self.device)

        kp_detector = KPDetector(**config["model_params"]["kp_detector_params"], **config["model_params"]["common_params"])
        kp_detector.to(self.device)

        he_estimator = HEEstimator(**config["model_params"]["he_estimator_params"], **config["model_params"]["common_params"])
        he_estimator.to(self.device)

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        generator.load_state_dict(checkpoint["generator"])
        kp_detector.load_state_dict(checkpoint["kp_detector"])
        he_estimator.load_state_dict(checkpoint["he_estimator"])

        generator.eval()
        kp_detector.eval()
        he_estimator.eval()
        
        return generator, kp_detector, he_estimator

    def reset_frames(self):
        self.kp_driving_initial = None
        self.coordinates = None

    def set_source_image(self, source_image):
        # FOMM Expects 256x256
        source_image_res = cv2.resize(source_image, (256, 256))
        self.source = to_tensor(source_image_res).to(self.device)
        if self.device == 'cuda':
            self.source = self.source.half()
        
        with torch.no_grad():
            self.kp_canonical = self.kp_detector(self.source.float() if self.device == 'cuda' else self.source)
            self.he_source = self.he_estimator(self.source.float() if self.device == 'cuda' else self.source)
            
            if self.kp_driving_initial is not None and getattr(self, 'coordinates', None) is not None:
                self.kp_source = keypoint_transformation(
                    self.kp_canonical, self.he_source, free_view=True, 
                    yaw=self.coordinates["yaw"], pitch=self.coordinates["pitch"], roll=self.coordinates["roll"]
                )
            else:
                self.kp_source = None 

    def predict(self, driving_frame):
        assert self.source is not None, "call set_source_image()"
        
        driving_frame_res = cv2.resize(driving_frame, (256, 256))
        driving = to_tensor(driving_frame_res).to(self.device)

        with torch.no_grad():
            he_driving = self.he_estimator(driving)
            
            if self.kp_driving_initial is None:
                self.kp_driving_initial, self.coordinates = keypoint_transformation(self.kp_canonical, he_driving, output_coord=True)
                self.kp_source = keypoint_transformation(
                    self.kp_canonical, self.he_source, free_view=True, 
                    yaw=self.coordinates["yaw"], pitch=self.coordinates["pitch"], roll=self.coordinates["roll"]
                )
                self.start_frame = driving_frame.copy()
                self.start_frame_kp = self.get_frame_kp(driving_frame)
            elif self.kp_source is None and getattr(self, 'coordinates', None) is not None:
                self.kp_source = keypoint_transformation(
                    self.kp_canonical, self.he_source, free_view=True, 
                    yaw=self.coordinates["yaw"], pitch=self.coordinates["pitch"], roll=self.coordinates["roll"]
                )

            kp_driving = keypoint_transformation(self.kp_canonical, he_driving)
            
            kp_norm = normalize_kp_fvv(
                kp_source=self.kp_source,
                kp_driving=kp_driving,
                kp_driving_initial=self.kp_driving_initial,
                use_relative_movement=self.relative,
                adapt_movement_scale=self.adapt_movement_scale,
            )

            out = self.generator(self.source, kp_source=self.kp_source, kp_driving=kp_norm, fp16=(self.device == 'cuda'))
            image = np.transpose(out["prediction"].data.cpu().numpy(), [0, 2, 3, 1])[0]
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
            
            # Resize back to driving frame size if needed, but Avatarify usually expects 512x512
            # Let's return the 256x256 result as it is usually enough and faster
            return image

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
