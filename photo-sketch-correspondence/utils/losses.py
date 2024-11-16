import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskMSE(nn.Module):
    """
    A masked version of MSE, designed for flow.
    It only cares about valid pixels, and ignores errors at pixels with out-of-bound values.
    Ignore conditions: value == 0: typically caused by zero padding in grid_sample
                       value > 1.0 or value < -1.0: typically happened when flow indices went out of range

    """
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        mse = F.mse_loss(input, target, reduction="none")

        mask = torch.ones(mse.shape, dtype=torch.bool).to(mse.device)
        input_mask = input != 0
        target_mask = target != 0
        mask = mask & input_mask & target_mask

        input_mask = torch.abs(input) <= 1
        target_mask = torch.abs(target) <= 1
        mask = mask & input_mask & target_mask
        mask = mask.detach().flatten()

        mse = mse.flatten()
        return mse[mask].mean()

########## 0617 epipolar constraint loss 추가     
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class MaskMSE_epi(nn.Module):
    def __init__(self):
        super().__init__()

    def estimate_fundamental_matrix(self, inputs, targets):
        N, H, W, _ = inputs.shape
        inputs_reshaped = inputs.reshape(-1, 2)
        targets_reshaped = targets.reshape(-1, 2)

        inputs_np = inputs_reshaped.detach().cpu().numpy()
        targets_np = targets_reshaped.detach().cpu().numpy()
        
        F, mask = cv2.findFundamentalMat(inputs_np, targets_np, cv2.FM_RANSAC)
        
        # 이 에러는 cv2.findFundamentalMat 함수가 유효한 Fundamental Matrix 𝐹
        # F를 찾지 못했을 때 발생합니다. 이는 주어진 입력 데이터가 충분히 좋지 않거나, 
        # 대응점이 충분하지 않기 때문일 수 있습니다. 이러한 경우를 처리하기 위해,
        # F가 None일 경우 기본값으로 단위 행렬을 반환하도록 코드를 수정할 수 있습니다.
        
        if F is None:
            # Return an identity matrix if F is not found
            F = np.eye(3)
        
        return torch.tensor(F, dtype=torch.float32, device=inputs.device)

    def forward(self, inputs, targets):
        # Compute MSE loss with masking
        mse = F.mse_loss(inputs, targets, reduction="none")

        mask = torch.ones(mse.shape, dtype=torch.bool).to(mse.device)
        input_mask = inputs != 0
        target_mask = targets != 0
        mask = mask & input_mask & target_mask

        input_mask = torch.abs(inputs) <= 1
        target_mask = torch.abs(targets) <= 1
        mask = mask & input_mask & target_mask
        mask = mask.detach().flatten()

        mse = mse.flatten()
        mse_loss = mse[mask].mean()

        # Estimate Fundamental Matrix F
        F_matrix = self.estimate_fundamental_matrix(inputs, targets)

        # Convert points to homogeneous coordinates
        inputs_reshaped = inputs.reshape(-1, 2)
        targets_reshaped = targets.reshape(-1, 2)
        
        p1_hom = torch.cat([inputs_reshaped, torch.ones(inputs_reshaped.shape[0], 1).to(inputs.device)], dim=1)
        p2_hom = torch.cat([targets_reshaped, torch.ones(targets_reshaped.shape[0], 1).to(targets.device)], dim=1)

        # Compute Epipolar Constraint loss
        p2_F_p1 = torch.einsum('bi,ij,bj->b', p2_hom, F_matrix, p1_hom)
        epi_loss = torch.mean(p2_F_p1 ** 2)

        # Combine losses
        total_loss = mse_loss + epi_loss
        return total_loss
