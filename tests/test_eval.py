from eval.run import ADVICE_WORDS


def test_advice_detector_ignores_client_names():
    assert not ADVICE_WORDS.search("Key partnerships with Best Buy and ABB; cross-sell opportunities")
    for phrase in ("We maintain a buy rating", "price target of 3,500", "we recommend buying the shares",
                   "Sell the stock", "upgrade to overweight"):
        assert ADVICE_WORDS.search(phrase), phrase
