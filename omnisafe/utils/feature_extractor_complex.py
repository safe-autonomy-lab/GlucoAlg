import torch
import torch.nn as nn

class SpatialCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(SpatialCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, out_channels, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(2, padding=1),
            nn.Flatten()
        )
    
    def forward(self, x):
        # Input x: [bs, np, channels, seq_len], e.g., [4, 60, 5, 6]
        bs, np, channels, seq_len = x.shape
        # Reshape to [bs * np, channels, seq_len] for CNN
        x = x.reshape(bs * np, channels, seq_len)
        # Extract features: [bs * np, out_features]
        features = self.cnn(x)
        # Reshape back to [bs, np, out_features]
        features = features.view(bs, np, -1)
        # Aggregate within each environment (e.g., mean over pedestrians)
        env_features = features.mean(dim=1)  # [bs, out_features]
        return env_features

class VisibleMaskCNN(nn.Module):
    def __init__(self, num_pedestrians, out_features):
        super(VisibleMaskCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(32 * (num_pedestrians // 4), out_features)  # e.g., 32 * (60 // 4) = 32 * 15 = 480
        )
    
    def forward(self, x):
        # Input x: [bs, np], e.g., [4, 60]
        x = x.unsqueeze(1)  # [bs, 1, np], e.g., [4, 1, 60]
        features = self.cnn(x)  # [bs, out_features], e.g., [4, 64]
        return features

class FeatureExtractor(nn.Module):
    def __init__(self, hidden_dim=64):
        super(FeatureExtractor, self).__init__()
        self.hidden_dim = hidden_dim
        self.visible_mask_cnn = VisibleMaskCNN(num_pedestrians=60, out_features=hidden_dim)
        # MLPs remain unchanged
        self.robot_mlp = nn.Sequential(
            nn.Linear(7 + 2 + hidden_dim, hidden_dim),  # 7 + 2 + 64
            nn.ReLU()
        )
        self.scalar_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU()
        )
        # Define Conv1D layers
        self.spatial_cnn = SpatialCNN(in_channels=2, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.ped_conv = SpatialCNN(in_channels=5, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.veh_conv = SpatialCNN(in_channels=5, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.robot_history_conv = SpatialCNN(in_channels=5, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.robot_plan_conv = SpatialCNN(in_channels=5, out_channels=hidden_dim, kernel_size=3, padding=1)

    def forward(self, obs_dict, batch_size=4):
        # Process visible_masks
        visible_mask_features = self.visible_mask_cnn(obs_dict['visible_masks'])  # [4, hidden_dim]

        # Process robot_node, temporal_edges, and visible_mask_features
        robot_input = torch.cat([
            obs_dict['robot_node'].squeeze(1),      # [4, 7]
            obs_dict['temporal_edges'].squeeze(1),  # [4, 2]
            visible_mask_features        # [4, hidden_dim]
        ], dim=-1)  # [4, 7 + 2 + hidden_dim]
        robot_features = self.robot_mlp(robot_input)  # [4, hidden_dim]

        # Process detected_human_num
        scalar_input = obs_dict['detected_human_num']  # [4, 1]
        scalar_features = self.scalar_mlp(scalar_input)  # [4, hidden_dim]

        # Process spatial_edges with SpatialCNN
        spatial_features = self.spatial_cnn(obs_dict['spatial_edges'].reshape(-1, 60, 2, 7))  # [4, hidden_dim], assuming [4, 60, 2, 7]

        # Process ped_pos with SpatialCNN
        ped_pos = obs_dict['ped_pos'].permute(0, 2, 3, 1)  # [4, 60, 5, 6]
        ped_features = self.ped_conv(ped_pos)  # [4, hidden_dim]

        # Process veh_pos with SpatialCNN
        veh_pos = obs_dict['veh_pos'].permute(0, 2, 3, 1)  # [4, 15, 5, 12]
        veh_features = self.veh_conv(veh_pos)  # [4, hidden_dim]

        # Process robot_pos with SpatialCNN
        robot_pos = obs_dict['robot_pos'].squeeze(2).permute(0, 2, 1).unsqueeze(1)  # [4, 1, 5, 6]
        robot_pos_features = self.robot_history_conv(robot_pos)  # [4, hidden_dim]

        # Process robot_plan with SpatialCNN
        robot_plan = obs_dict['robot_plan'].squeeze(2).permute(0, 2, 1).unsqueeze(1)  # [4, 1, 5, 6]
        robot_plan_features = self.robot_plan_conv(robot_plan)  # [4, hidden_dim]

        # Combine all features
        combined_features = torch.cat([
            robot_features,         # [4, hidden_dim]
            scalar_features,        # [4, hidden_dim]
            ped_features,           # [4, hidden_dim]
            veh_features,           # [4, hidden_dim]
            spatial_features,       # [4, hidden_dim]
            robot_pos_features + robot_plan_features  # [4, hidden_dim]
        ], dim=1)  # [4, hidden_dim * 6]

        return combined_features