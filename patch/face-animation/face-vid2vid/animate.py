import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import imageio
from scipy.spatial import ConvexHull
import numpy as np

from sync_batchnorm import DataParallelWithCallback

def regularize_jacobian(jacobian, stabilization=1.0):
    if stabilization <= 0: return jacobian
    U, S, V = torch.svd(jacobian)
    S_mean = S.mean(dim=-1, keepdim=True).repeat(1, 1, 2)
    S_new = (1.0 - stabilization) * S + stabilization * S_mean
    R = torch.matmul(U, torch.matmul(torch.diag_embed(S_new), V.transpose(-1, -2)))
    det = torch.det(R)
    V_fixed = V.clone()
    V_fixed[..., 1] *= torch.sign(det).unsqueeze(-1)
    return torch.matmul(U, torch.matmul(torch.diag_embed(S_new), V_fixed.transpose(-1, -2)))

def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale_flag=False,
                 use_relative_movement=False, use_relative_jacobian=False, **kwargs):
    
    adapt_movement_scale = 1.0
    if adapt_movement_scale_flag:
        source_area = (kp_source['value'].max(dim=1)[0] - kp_source['value'].min(dim=1)[0]).prod()
        driving_area = (kp_driving['value'].max(dim=1)[0] - kp_driving['value'].min(dim=1)[0]).prod()
        adapt_movement_scale = np.sqrt(source_area) / (np.sqrt(driving_area) + 1e-6)
        adapt_movement_scale = np.clip(adapt_movement_scale, 0.95, 1.05)

    kp_new = {k: v.clone() for k, v in kp_driving.items()}
    mouth_indices = kwargs.get('mouth_kp_indices', [])
    eye_indices = kwargs.get('eye_kp_indices', [])
    eyebrow_indices = kwargs.get('eyebrow_kp_indices', [])
    jaw_indices = kwargs.get('jaw_kp_indices', [])
    nose_indices = kwargs.get('nose_kp_indices', [])
    cheek_indices = kwargs.get('cheek_kp_indices', [])
    neck_indices = kwargs.get('neck_kp_indices', [])
    
    face_indices = (mouth_indices if mouth_indices else []) + \
                   (eye_indices if eye_indices else []) + \
                   (eyebrow_indices if eyebrow_indices else []) + \
                   (jaw_indices if jaw_indices else []) + \
                   (nose_indices if nose_indices else []) + \
                   (cheek_indices if cheek_indices else []) + \
                   (neck_indices if neck_indices else [])
    
    all_indices = set(range(kp_driving['value'].shape[1]))
    background_indices = list(all_indices - set(face_indices))

    if use_relative_movement:
        if face_indices:
            driving_centroid = kp_driving['value'][:, face_indices].mean(dim=1, keepdim=True)
            initial_centroid = kp_driving_initial['value'][:, face_indices].mean(dim=1, keepdim=True)
            global_delta = (driving_centroid - initial_centroid)
            local_delta = (kp_driving['value'] - driving_centroid) - (kp_driving_initial['value'] - initial_centroid)
            
            # REHAUL: Calculate distance-based falloff for global movement
            dist_to_centroid = torch.norm(kp_driving['value'] - driving_centroid, dim=2, keepdim=True)
            face_spread = torch.std(kp_driving['value'][:, face_indices], dim=1, keepdim=True).mean(dim=-1, keepdim=True)
            face_radius = face_spread * 2.5
            
            # Radial falloff: 1.0 at center, fades to 0.0 at 2x radius
            radial_falloff = torch.clamp(1.0 - (dist_to_centroid - face_radius) / (face_radius * 1.5), 0.0, 1.0)
            
            turn_intensity = torch.abs(global_delta[:, :, 0:1]) * 1.0
            turn_factor = torch.clamp(torch.exp(-turn_intensity), 0.8, 1.0)
            
            bg_mask = torch.zeros(kp_driving['value'].shape[1], device=kp_driving['value'].device).view(1, -1, 1)
            bg_mask[0, background_indices] = 1.0
            
            expression_mask = torch.zeros(kp_driving['value'].shape[1], device=kp_driving['value'].device).view(1, -1, 1)
            expression_mask[0, mouth_indices + jaw_indices + eyebrow_indices + eye_indices] = 1.0
            
            nose_mask = torch.zeros(kp_driving['value'].shape[1], device=kp_driving['value'].device).view(1, -1, 1)
            if nose_indices:
                nose_mask[0, nose_indices] = 1.0
            
            cheek_mask = torch.zeros(kp_driving['value'].shape[1], device=kp_driving['value'].device).view(1, -1, 1)
            if cheek_indices:
                cheek_mask[0, cheek_indices] = 1.0

            neck_mask = torch.zeros(kp_driving['value'].shape[1], device=kp_driving['value'].device).view(1, -1, 1)
            if neck_indices:
                neck_mask[0, neck_indices] = 1.0
            
            # Mouth Sensitivity: Boost mouth movement for smaller characters
            sensitivity = kwargs.get('mouth_sensitivity', 1.0)
            mouth_mask = torch.zeros(kp_driving['value'].shape[1], device=kp_driving['value'].device).view(1, -1, 1)
            if mouth_indices:
                mouth_mask[0, mouth_indices] = 1.0
            
            # Anti-Elasticity: Penalize local_delta if it exceeds 15% of face radius
            face_radius_val = torch.std(kp_driving['value'][:, face_indices], dim=1, keepdim=True).mean(dim=-1, keepdim=True)
            local_magnitude = torch.norm(local_delta, dim=-1, keepdim=True)
            elasticity_limit = face_radius_val * 0.4 # Limit movement to 40% of face spread
            elasticity_dampener = torch.where(local_magnitude > elasticity_limit, elasticity_limit / (local_magnitude + 1e-6), torch.ones_like(local_magnitude))
            local_delta = local_delta * elasticity_dampener
            
            # Neck stabilization: 50% responsive
            # Nose stabilization: 100% responsive
            # Cheeks: 80% responsive
            # Expressions: 98% responsive * sensitivity
            # Background: 0% responsive
            move_factor = (1.0 - bg_mask) * (neck_mask * 0.6 + nose_mask * 1.0 + cheek_mask * 0.8 + (1.0 - nose_mask - cheek_mask - neck_mask) * (expression_mask * 0.98 + (1.0 - expression_mask) * turn_factor)) + (bg_mask * 0.0)
            
            # Apply Mouth Sensitivity
            local_delta = local_delta * (1.0 + mouth_mask * (sensitivity - 1.0))
            
            # REHAUL: Global movement intensity follows the radial falloff
            global_intensity = radial_falloff * (1.0 - bg_mask)
            kp_value_diff = (global_delta * global_intensity) + (local_delta * move_factor)
        else:
            kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
            kp_value_diff[:, :, 0] *= 0.8 # Dampen horizontal sway
            
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            
            if face_indices:
                head_jac = jacobian_diff[:, face_indices].mean(dim=1, keepdim=True)
                U, S, V = torch.svd(head_jac)
                # Force Pure Rotation for the head (Isotropic Head)
                head_jac = torch.matmul(U, torch.matmul(torch.diag_embed(torch.ones_like(S)), V.transpose(-1, -2)))
                
                bg_mask_jac = torch.zeros(jacobian_diff.shape[1], device=jacobian_diff.device).view(1, -1, 1, 1)
                bg_mask_jac[0, background_indices] = 1.0
                
                neck_mask_jac = torch.zeros(jacobian_diff.shape[1], device=jacobian_diff.device).view(1, -1, 1, 1)
                if neck_indices:
                    neck_mask_jac[0, neck_indices] = 1.0
                    
                jacobian_diff = (1.0 - bg_mask_jac - neck_mask_jac) * jacobian_diff + (neck_mask_jac * 0.5 * (jacobian_diff + torch.eye(2, device=jacobian_diff.device).view(1, 1, 2, 2))) + bg_mask_jac * torch.eye(2, device=jacobian_diff.device).view(1, 1, 2, 2)

            stabilization = kwargs.get('jacobian_stabilization', 0.5)
            dampening = kwargs.get('jacobian_dampening', 0.95) # Even higher dampening (95%)
            
            jacobian_diff = regularize_jacobian(jacobian_diff, stabilization=stabilization)

            U, S, V = torch.svd(jacobian_diff)
            S_mean = S.mean(dim=-1, keepdim=True).repeat(1, 1, 2)
            S_dampened = (1.0 - dampening) * S + dampening * S_mean
            
            # Prevent extreme stretching (>1.03x or <0.97x) to stop face from deforming when turning
            S_min, S_max = 0.97, 1.03
            S_dampened = torch.clamp(S_dampened, S_min, S_max)
            
            jacobian_diff = torch.matmul(U, torch.matmul(torch.diag_embed(S_dampened), V.transpose(-1, -2)))

            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new
