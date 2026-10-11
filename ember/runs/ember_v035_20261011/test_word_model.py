"""Run directly to verify the released model and integer word heads."""
import hashlib
from word_model import EmberWord


def main():
    model = EmberWord()  # verifies both model hashes before loading
    assert len(model.words) == 8192
    assert model.W["next"].shape == (8192, 240)
    probe = model.features("The weather today is ").astype("int8")
    assert hashlib.sha256(probe.tobytes()).hexdigest() == (
        "7c9e961e1d02971d8f181ae457e907b3e18e8b945aef62cc5fc58cbc01d75cf7"
    )
    assert model.suggest("Thank you for ", top_k=1) == ['your']
    assert model.suggest("Thank you for be", prefix2=True, top_k=1) == ['being']
    for word in model.suggest("The weather today is su", prefix2=True, top_k=5):
        assert word.startswith("su")
    assert model.suggest("The weather today is zx", prefix2=True) == []
    print("Ember v0.3.5 model hashes, scalar probe and word suggestions: OK")


if __name__ == "__main__":
    main()
