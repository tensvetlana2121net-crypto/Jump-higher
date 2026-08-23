import pytest
from fastapi.routing import APIRoute

from jumpbot.api.miniapp import make_jump_type, router
from jumpbot.config import get_settings


def test_miniapp_routes_are_registered_before_static_mount() -> None:
    api_routes = {
        (route.path, method)
        for route in router.routes
        if isinstance(route, APIRoute)
        for method in route.methods
    }

    assert ("/miniapp/api/me", "GET") in api_routes
    assert ("/miniapp/api/analyses", "GET") in api_routes
    assert ("/miniapp/api/analyses", "POST") in api_routes
    assert ("/miniapp/api/analyses/{jump_id}", "GET") in api_routes


def test_miniapp_static_assets_exist() -> None:
    static_dir = get_settings().miniapp_static_dir

    assert (static_dir / "index.html").is_file()
    assert (static_dir / "styles.css").is_file()
    assert (static_dir / "app.js").is_file()
    assert (static_dir / "classification.css").is_file()


@pytest.mark.parametrize(
    ("name", "rotations", "expected"),
    [
        ("axel", 1, "1_axel"),
        ("loop", 2, "2_loop"),
        ("salchow", 3, "3_salchow"),
        ("flip", 4, "4_flip"),
        ("lutz", 3, "3_lutz"),
        ("toe_loop", 2, "2_toe_loop"),
    ],
)
def test_jump_classification(name: str, rotations: int, expected: str) -> None:
    assert make_jump_type(name, rotations) == expected


@pytest.mark.parametrize(("name", "rotations"), [("unknown", 2), ("axel", 5)])
def test_jump_classification_rejects_unknown_values(name: str, rotations: int) -> None:
    with pytest.raises(ValueError):
        make_jump_type(name, rotations)
