import os
import torch
from protconv.models.simple_cnn import Simple1DCNN
from protconv.data.loader import AA_TO_IDX, UNK_INDEX
from protconv.utils.geometry import ca_trace_reconstruction_torch
from protconv.utils.pdb import fragment_to_pdb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASELINE_MODEL_PATH = os.path.join(os.getcwd(), "models/baseline.pt")


def encode_sequence(seq):
    """Encode a sequence using the robust AA mapping."""
    return torch.tensor([AA_TO_IDX.get(aa, UNK_INDEX) for aa in seq], dtype=torch.long)


def predict(
    seq, model_class=Simple1DCNN, model_path=BASELINE_MODEL_PATH, vocab_size=21
):
    """Predict CA trace for a sequence using the specified model class and checkpoint."""
    model = model_class(vocab_size=vocab_size)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    encoded = encode_sequence(seq).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_vectors = model(encoded)[0]
        mask = torch.ones(len(seq), dtype=torch.bool, device=DEVICE)
        ca_trace = ca_trace_reconstruction_torch(pred_vectors, mask)
    return ca_trace.cpu().numpy()


def main(seq, model_path=BASELINE_MODEL_PATH):
    ca_trace = predict(seq, model_path)

    # Save to pdb
    pdb_path = os.path.join(os.getcwd(), "output.pdb")
    fragment_to_pdb({"primary": args.sequence, "tertiary": ca_trace}, pdb_path)
    print(f"Saved to {pdb_path}")
