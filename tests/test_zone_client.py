from unittest.mock import MagicMock, patch

from src.zone_client import ZoneClient

ZONES = [
    {"zone_id": "restricted_1", "points": [[0, 0], [10, 0], [10, 10]], "allowed_workers": ["W001"]},
]


def test_upload_posts_zones_and_returns_true():
    with patch("src.zone_client.requests.post") as post:
        post.return_value = MagicMock(status_code=200, raise_for_status=lambda: None)
        client = ZoneClient("http://server/api/zones")

        assert client.upload_zones(ZONES) is True
        post.assert_called_once()
        args, kwargs = post.call_args
        assert args[0] == "http://server/api/zones"
        assert kwargs["json"]["zones"] == ZONES
        assert kwargs["json"]["count"] == 1


def test_upload_disabled_skips_network():
    with patch("src.zone_client.requests.post") as post:
        client = ZoneClient("http://server/api/zones", enabled=False)
        assert client.upload_zones(ZONES) is False
        post.assert_not_called()


def test_upload_swallows_network_error():
    with patch("src.zone_client.requests.post", side_effect=Exception("connection refused")) as post:
        client = ZoneClient("http://server/api/zones")
        # Must not raise — local save already happened in the caller.
        assert client.upload_zones(ZONES) is False
        post.assert_called_once()


def test_upload_raises_for_http_error_returns_false():
    def raise_http():
        raise Exception("500 Server Error")

    with patch("src.zone_client.requests.post") as post:
        post.return_value = MagicMock(raise_for_status=raise_http)
        client = ZoneClient("http://server/api/zones")
        assert client.upload_zones(ZONES) is False
