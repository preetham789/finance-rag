import pytest
from fastapi import HTTPException

from src import security as sec


def test_parse_api_keys_hashes_and_validates():
    parsed = sec.parse_api_keys("k1:admin, k2:analyst")
    assert set(parsed.values()) == {"admin", "analyst"}
    assert all(len(h) == 64 for h in parsed)
    assert "k1" not in parsed
    with pytest.raises(ValueError):
        sec.parse_api_keys("k1:superuser")
    with pytest.raises(ValueError):
        sec.parse_api_keys("nocolon")


def test_get_principal_paths(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "apikey")
    monkeypatch.setenv("API_KEYS", "good:analyst")
    assert sec.get_principal("good").role == "analyst"
    for bad in (None, "", "wrong"):
        with pytest.raises(HTTPException) as e:
            sec.get_principal(bad)
        assert e.value.status_code == 401
    monkeypatch.setenv("API_KEYS", "")
    with pytest.raises(HTTPException) as e:
        sec.get_principal("good")
    assert e.value.status_code == 503


def test_auth_off_runs_as_admin(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "off")
    assert sec.get_principal(None).role == "admin"


def test_scope_merges_role_and_request():
    analyst = sec.Principal("x", "analyst")
    assert sec.build_scope(analyst) == {"doc_type": ["rbi", "company"]}
    assert sec.build_scope(analyst, company="TCS", doc_type="company") == {
        "doc_type": ["company"],
        "company": ["TCS"],
    }
    assert sec.build_scope(sec.Principal("x", "admin")) == {}


def test_scope_denies_out_of_role_requests_and_unknown_roles():
    with pytest.raises(HTTPException) as e:
        sec.build_scope(sec.Principal("x", "public"), doc_type="company")
    assert e.value.status_code == 403
    with pytest.raises(HTTPException) as e:
        sec.build_scope(sec.Principal("x", "ghost"))
    assert e.value.status_code == 403


def test_chroma_where_shapes():
    assert sec.scope_to_chroma_where({}) is None
    assert sec.scope_to_chroma_where({"doc_type": ["rbi"]}) == {
        "doc_type": {"$in": ["rbi"]}
    }
    both = sec.scope_to_chroma_where({"doc_type": ["rbi"], "company": ["RBI"]})
    assert both == {
        "$and": [{"doc_type": {"$in": ["rbi"]}}, {"company": {"$in": ["RBI"]}}]
    }


def test_sparse_side_filter_agrees_with_scope():
    scope = {"doc_type": ["rbi"]}
    assert sec.scope_allows(scope, {"doc_type": "rbi", "company": "RBI"})
    assert not sec.scope_allows(scope, {"doc_type": "company", "company": "TCS"})
    assert not sec.scope_allows(scope, {})  # missing metadata fails closed
    assert sec.scope_allows({}, {"anything": 1})


def test_sliding_window_limiter():
    now = [0.0]
    lim = sec.SlidingWindowLimiter(3, 60, clock=lambda: now[0])
    assert [lim.hit("k")[0] for _ in range(3)] == [True, True, True]
    allowed, retry = lim.hit("k")
    assert not allowed and 0 < retry <= 60
    assert lim.hit("other")[0]  # separate key, separate budget
    now[0] = 61.0
    assert lim.hit("k")[0]
