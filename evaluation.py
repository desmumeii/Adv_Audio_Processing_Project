import torch

from model import ClassifierModel


def main():
    """Main evaluation function."""
    # Determine device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'Process on {device}', end='\n\n')

    # Initialize model
    model = ClassifierModel()
    checkpoint = torch.load("best_model.pt")

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # TODO: Add evaluation code here
    # Check model.predict() method in model.py for reference on how to get predictions from the model.


if __name__ == "__main__":
    main()