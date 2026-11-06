from app.db import session


def test_ssl_connect_args(monkeypatch):
    s = session.get_settings()
    monkeypatch.setattr(s, "database_url_override", None)
    monkeypatch.setattr(s, "mysql_ssl", False)
    monkeypatch.setattr(s, "mysql_ssl_ca_cert", None)
    assert session._connect_args() == {}

    monkeypatch.setattr(s, "mysql_ssl", True)
    assert session._connect_args() == {"ssl": {"verify_mode": "none"}}

    pem = "-----BEGIN CERTIFICATE-----\\nABC\\n-----END CERTIFICATE-----"  # pasted as one line
    monkeypatch.setattr(s, "mysql_ssl_ca_cert", pem)
    args = session._connect_args()
    with open(args["ssl"]["ca"], encoding="utf-8") as f:
        assert f.read().splitlines() == ["-----BEGIN CERTIFICATE-----", "ABC", "-----END CERTIFICATE-----"]
