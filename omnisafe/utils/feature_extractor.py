import torch
import torch.nn as nn

class FeatureExtractor2(nn.Module):
    def __init__(self, hidden_dim=64, seq_len=6, input_dim=4, num_layers=1):
        super(FeatureExtractor2, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len  # Number of past timesteps in robot_pos
        self.input_dim = input_dim  # 4 features: [x, y, vx, vy]

        # LSTM to process the robot_pos sequence
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Attention mechanism for LSTM outputs (optional)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )

        # MLP for robot_node and temporal_edges
        self.robot_mlp = nn.Sequential(
            nn.Linear(7 + 2 + 2, hidden_dim),  # 7 from robot_node, 2 from temporal_edges
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Final MLP to combine all features
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Combine robot MLP and LSTM outputs
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, obs_dict, batch_size=4):
            # Keep batch dimension
        robot_node = obs_dict['robot_node']  # [batch_size, 7], e.g., [4, 7] or [1, 7]
        temporal_edges = obs_dict['temporal_edges']  # [batch_size, 2], e.g., [4, 2] or [1, 2]
        goal_pos = obs_dict['goal_pos']  # [batch_size, 2], e.g., [4, 2] or [1, 2]
        robot_input = torch.cat([robot_node, temporal_edges, goal_pos], dim=-1)  # [batch_size, 11]
        robot_features = self.robot_mlp(robot_input)  # [batch_size, 64]

        # Process sequence data (unchanged)
        robot_pos = obs_dict['robot_pos'].squeeze(2)  # [batch_size, 6, 5]
        robot_pos = robot_pos[:, :, :4]  # [batch_size, 6, 4]
        lstm_out, _ = self.lstm(robot_pos)  # [batch_size, 6, 64]
        attn_weights = self.attention(lstm_out)  # [batch_size, 6, 1]
        attn_out = torch.sum(lstm_out * attn_weights, dim=1)  # [batch_size, 64]

        # Concatenate with consistent dimensions
        combined_features = torch.cat([robot_features.squeeze(1), attn_out], dim=1)  # [batch_size, 128]
        output = self.final_mlp(combined_features)  # [batch_size, 64]
        return output



class FeatureExtractor(nn.Module):
    def __init__(self, hidden_dim=64, seq_len=6, input_dim=4, num_layers=1):
        super(FeatureExtractor, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len  # Number of past timesteps in robot_pos
        self.input_dim = input_dim  # 4 features: [x, y, vx, vy]

        # MLP for robot_node and temporal_edges
        self.robot_mlp = nn.Sequential(
            nn.Linear(7 + 2, hidden_dim),  # 7 from robot_node, 2 from temporal_edges
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Final MLP to combine all features
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, obs_dict, batch_size=4):
        robot_node = obs_dict['robot_node']  # [batch_size, 7], e.g., [4, 7] or [1, 7]
        temporal_edges = obs_dict['temporal_edges']  # [batch_size, 2], e.g., [4, 2] or [1, 2]
        robot_input = torch.cat([robot_node, temporal_edges], dim=-1)  # [batch_size, 9]
        robot_features = self.robot_mlp(robot_input)  # [batch_size, 64]

        output = self.final_mlp(robot_features)  # [batch_size, 64]
        return output
    
# Example usage
if __name__ == "__main__":
    # Sample observation
    obs = {
        'robot_node': torch.randn(1, 7),
        'temporal_edges': torch.randn(1, 2),
        'robot_pos': torch.randn(1, 6, 1, 5),  # [1, 6, 1, 5]
    }
    extractor = FeatureExtractor(hidden_dim=64)
    features = extractor(obs)
    print(f"Feature vector shape: {features.shape}")  # Should be [64]
    print(f"Features: {features}")