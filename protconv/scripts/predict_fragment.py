from protconv.models.simple_cnn import Simple1DCNN
import torch
from protconv.data.loader import AA_TO_IDX, UNK_INDEX
from protconv.utils.geometry import ca_trace_reconstruction_torch
from protconv.utils.pdb import fragment_to_pdb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASELINE_MODEL_PATH = os.path.join(os.getcwd(), "models/baseline.pt")


def encode_sequence(seq):
    """Encode a sequence using the robust AA mapping."""
    return torch.tensor([AA_TO_IDX.get(aa, UNK_INDEX) for aa in seq], dtype=torch.long)


def predict(seq, model_path):
    # Load model
    model = Simple1DCNN(vocab_size=21)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Encode sequence
    seq_tensor = encode_sequence(seq).unsqueeze(0).to(DEVICE)  # (1, seq_len)
    mask = torch.ones(seq_tensor.shape, dtype=torch.bool, device=DEVICE)  # (1, seq_len)

    with torch.no_grad():
        pred_vectors = model(seq_tensor)[0]  # (seq_len, 6)
        ca_trace = ca_trace_reconstruction_torch(pred_vectors, mask[0])  # (seq_len, 3)

    return ca_trace


def main(seq, model_path=BASELINE_MODEL_PATH):
    ca_trace = predict(seq, model_path)

    # Save to pdb
    pdb_path = os.path.join(os.getcwd(), "output.pdb")
    fragment_to_pdb({"primary": seq, "tertiary": ca_trace.cpu().numpy()}, pdb_path)
    print(f"Saved to {pdb_path}")


if __name__ == "__main__":
    raise NotImplementedError("Call script with `python -m protconv train`")
