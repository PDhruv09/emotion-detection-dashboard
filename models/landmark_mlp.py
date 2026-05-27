import torch.nn as nn


class LandmarkEmotionMLP(nn.Module):
    def __init__(
        self,
        input_size,
        num_classes=7,
        hidden_size=384,
        dropout=0.35,
        num_layers=3,
        architecture="modern",
    ):
        super().__init__()

        if architecture == "legacy":
            self.network = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.BatchNorm1d(hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, num_classes),
            )
            return

        layers = []
        current_size = input_size
        layer_size = hidden_size

        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(current_size, layer_size),
                    nn.LayerNorm(layer_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            current_size = layer_size
            layer_size = max(layer_size // 2, 64)

        layers.append(nn.Linear(current_size, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
