"""Run directly to verify the released model, integer word readouts and word rule ticks."""
import hashlib
from word_model import EmberWord


def main():
    model = EmberWord()  # verifies both model hashes before loading
    assert len(model.words) == 32768 and model.K == 2 and model.T == 16 and model.mode == 'words'
    assert model.W["next"].shape == (32768, 816)
    probe = model.features("The weather today is ").astype("int8")
    assert hashlib.sha256(probe.tobytes()).hexdigest() == (
        "7c9e961e1d02971d8f181ae457e907b3e18e8b945aef62cc5fc58cbc01d75cf7"
    )
    for text, want in {'Thank you for ': ['your'], 'I would like to ': ['find']}.items():
        assert model.next_words(text, 1) == want, text
    for text, want in {'Thank you for be': ['being'], 'Thank you for b': ['being'], 'I would like to kno': ['know']}.items():
        assert model.complete(text, 1) == want, text
    for word in model.complete("The weather today is su", 5):
        assert word.startswith("su")
    assert model.complete("The weather today is zxq") == []
    assert model.suggest("Thank you for be", prefix2=True, top_k=1) == model.complete("Thank you for be", 1)
    print("Ember v0.3.8 model hashes, scalar probe, word rules and word suggestions: OK")


if __name__ == "__main__":
    main()
