import torch
from config import get_args

args = get_args()


class Feedforward(torch.nn.Module):
    """
    Feedforward neural network with configurable hidden layers.

    This network is designed for binary classification tasks with fairness constraints.
    It uses LeakyReLU activations and adds a sigmoid output layer for binary classification.
    """

    def __init__(self, input_size, hidden_sizes, num_classes, negative_slope=0.01, dropout=0.0):
        """
        Initialize the feedforward network.

        Args:
            input_size (int): Dimension of input features
            hidden_sizes (list of int): List of hidden layer sizes
            num_classes (int): Number of output classes (typically 1 for binary classification)
            negative_slope (float): Negative slope for LeakyReLU activation
            dropout (float): Dropout probability (0.0 means no dropout)

        Raises:
            ValueError: If input_size, hidden_sizes, or num_classes are invalid
        """
        super(Feedforward, self).__init__()

        # Validation
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if not hidden_sizes or not all(h > 0 for h in hidden_sizes):
            raise ValueError(f"All hidden_sizes must be positive, got {hidden_sizes}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        # Store architecture for reference
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.negative_slope = negative_slope
        self.dropout = dropout

        # Build network layers
        layers = []
        in_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(torch.nn.Linear(in_size, hidden_size))
            layers.append(torch.nn.LeakyReLU(negative_slope=negative_slope))

            # Add dropout if specified
            if dropout > 0.0:
                layers.append(torch.nn.Dropout(p=dropout))

            in_size = hidden_size

        # Output layer
        layers.append(torch.nn.Linear(in_size, num_classes))

        # Add sigmoid for binary classification
        if num_classes == 1:
            layers.append(torch.nn.Sigmoid())

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        return self.network(x)

    def get_architecture_info(self):
        """
        Get architecture information as a dictionary.

        Returns:
            dict: Dictionary containing network architecture details
        """
        return {
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'num_classes': self.num_classes,
            'negative_slope': self.negative_slope,
            'dropout': self.dropout,
            'total_parameters': sum(p.numel() for p in self.parameters())
        }

    def __repr__(self):
        """String representation of the model."""
        info = self.get_architecture_info()
        return (f"Feedforward(input={info['input_size']}, "
                f"hidden={info['hidden_sizes']}, "
                f"output={info['num_classes']}, "
                f"params={info['total_parameters']})")
