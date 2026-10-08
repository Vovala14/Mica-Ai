"""Run directly to verify the released model and integer word heads."""
import hashlib
from word_model import EmberWord


def main():
    model = EmberWord()  # verifies both model hashes before loading
    assert len(model.words) == 4339
    assert model.W["next"].shape == (4339, 240)
    probe = model.features("The weather today is ").astype("int8")
    assert hashlib.sha256(probe.tobytes()).hexdigest() == (
        "4b490a1b394589c421d31bc116a01e93bdfcfe3b66060a9471941110b9476f33"
    )
    assert model.suggest("Thank you for ", top_k=1) == ["being"]
    assert model.suggest("Thank you for be", prefix2=True, top_k=1) == ["being"]
    for word in model.suggest("The weather today is su", prefix2=True, top_k=5):
        assert word.startswith("su")
    assert model.suggest("The weather today is zx", prefix2=True) == []
    print("Ember v0.3.1 model hashes, scalar probe and word suggestions: OK")


if __name__ == "__main__":
    main()
