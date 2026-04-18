"""Tests for the FastAPI service contract."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from kg_backend.api import create_app


def test_health_and_stats(tiny_kg_path: Path) -> None:
    with TestClient(create_app(tiny_kg_path)) as client:
        health_response = client.get("/health")
        stats_response = client.get("/stats")

    assert health_response.status_code == 200
    assert health_response.json()["backend_loaded"] is True
    assert stats_response.status_code == 200
    assert stats_response.json()["num_triples"] == 7
    assert stats_response.json()["raw_num_triples"] == 8


def test_query_neighbors_contract(tiny_kg_path: Path) -> None:
    with TestClient(create_app(tiny_kg_path)) as client:
        response = client.post(
            "/query",
            json={
                "op": "get_neighbors",
                "entity_id": "alice",
                "relation_id": "likes",
                "direction": "out",
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "op": "get_neighbors",
        "entity_id": "alice",
        "relation_id": "likes",
        "direction": "out",
        "neighbors": ["bob", "carol"],
    }


def test_query_follow_path_contract(tiny_kg_path: Path) -> None:
    with TestClient(create_app(tiny_kg_path)) as client:
        response = client.post(
            "/query",
            json={
                "op": "follow_path",
                "start_entities": ["alice"],
                "path": [
                    {"relation_id": "likes", "direction": "out"},
                    {"relation_id": "parent", "direction": "in"},
                ],
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "op": "follow_path",
        "start_entities": ["alice"],
        "path": [
            {"relation_id": "likes", "direction": "out"},
            {"relation_id": "parent", "direction": "in"},
        ],
        "result_entities": ["carol"],
    }


def test_query_missing_relation_returns_404(tiny_kg_path: Path) -> None:
    with TestClient(create_app(tiny_kg_path)) as client:
        response = client.post(
            "/query",
            json={
                "op": "get_neighbors",
                "entity_id": "alice",
                "relation_id": "missing",
                "direction": "out",
            },
        )

    assert response.status_code == 404
    assert response.json()["error"] == "relation_not_found"
