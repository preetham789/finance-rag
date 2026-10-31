import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


def _loopback_available() -> bool:
    """Return True if we can open a TCP connection to 127.0.0.1 on an ephemeral port."""
    try:
        with socket.create_server(("127.0.0.1", 0)) as srv:
            host, port = srv.getsockname()
        with socket.create_connection((host, port), timeout=1):
            pass
    except OSError:
        return False
    return True


requires_loopback = pytest.mark.skipif(
    not _loopback_available(),
    reason="loopback TCP not available in this environment (WinError 10013 / sandbox)",
)


class MockAPI(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send(200, {"collection": "mock"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        q = body["question"].lower()
        if "repo" in q:  # right chunk is retrieved, but only at rank 2
            self._send(200, {
                "answer": "Cut to 6.25 per cent [Source: RBI, Page 112].",
                "sources": [
                    {"company": "TCS", "page_number": 5},
                    {"company": "RBI", "page_number": 112},
                ],
                "latency_ms": 900,
            })
        elif "employees" in q:
            self._send(200, {
                "answer": "TCS has 607,979 employees [Source: TCS, Page 24].",
                "sources": [{"company": "TCS", "page_number": 24}],
                "latency_ms": 650,
            })
        elif "apple" in q:
            self._send(200, {
                "answer": "The provided documents do not contain sufficient information.",
                "sources": [],
                "latency_ms": 700,
            })
        elif "future" in q:  # should refuse, hallucinates instead
            self._send(200, {
                "answer": "Revenue will reach 5 lakh crore [Source: TCS, Page 1].",
                "sources": [{"company": "TCS", "page_number": 1}],
                "latency_ms": 800,
            })
        else:
            self._send(500, {"detail": "boom"})


@pytest.fixture(scope="module")
def base_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockAPI)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


ROWS = [
    {"id": "repo", "category": "single_fact", "question": "repo rate?",
     "expect": {"sources": [{"company": "RBI", "pages": [112]}], "contains_all": ["6.25"]}},
    {"id": "emp", "category": "numeric", "question": "employees?",
     "expect": {"sources": [{"company": "TCS", "pages": [24]}], "contains_all": ["607979"]}},
    {"id": "apple", "category": "out_of_corpus", "question": "apple revenue?",
     "expect": {"refuse": True}},
    {"id": "future", "category": "temporal_trap", "question": "future revenue?",
     "expect": {"refuse": True}},
]


@requires_loopback
def test_scoring_end_to_end(eval_api, base_url):
    scored, skipped = eval_api.run_eval(base_url, ROWS, retries=0)
    agg = eval_api.aggregate(scored)
    by_id = {r["id"]: r for r in scored}
    assert skipped == 0
    assert by_id["repo"]["rank"] == 2
    assert agg["hit_rate"] == 1.0
    assert agg["mrr"] == 0.75  # (1/2 + 1/1) / 2
    assert agg["answer_ok_rate"] == 1.0  # "607,979" matches "607979"
    assert agg["refusal_accuracy"] == 0.5  # Apple refused, FY future trap hallucinated
    assert agg["citation_rate"] == 1.0
    assert agg["latency_p50_ms"] == 700  # nearest-rank p50 of [650, 700, 800, 900]


def test_page_tolerance_and_wrong_page(eval_api):
    srcs = [{"company": "RBI", "page_number": 113}]
    exp = [{"company": "RBI", "pages": [112]}]
    assert eval_api.first_hit_rank(srcs, exp, tol=0) is None
    assert eval_api.first_hit_rank(srcs, exp, tol=1) == 1
    assert eval_api.first_hit_rank(srcs, [{"company": "TCS", "pages": [113]}]) is None


@requires_loopback
def test_http_error_becomes_error_row_not_a_crash(eval_api, base_url):
    row = {"id": "err", "category": "x", "question": "unknown thing",
           "expect": {"contains_all": ["x"]}}
    scored, _ = eval_api.run_eval(base_url, [row], retries=0)
    assert scored[0]["error"]
    agg = eval_api.aggregate(scored)
    assert agg["errors"] == 1 and agg["answer_ok_rate"] is None


@requires_loopback
def test_skip_rows_are_not_scored(eval_api, base_url):
    rows = [{**ROWS[0], "skip": True}, ROWS[1]]
    scored, skipped = eval_api.run_eval(base_url, rows, retries=0)
    assert skipped == 1 and len(scored) == 1


def test_compare_gate(eval_api, tmp_path):
    def write(name, **overall):
        p = tmp_path / name
        p.write_text(json.dumps({"overall": overall}))
        return str(p)

    a = write("a.json", hit_rate=0.90, mrr=0.80, answer_ok_rate=0.90, refusal_accuracy=1.0)
    same = write("b.json", hit_rate=0.89, mrr=0.80, answer_ok_rate=0.91, refusal_accuracy=1.0)
    worse = write("c.json", hit_rate=0.80, mrr=0.80, answer_ok_rate=0.90, refusal_accuracy=1.0)
    assert eval_api.compare(a, same, 0.03) == 0
    assert eval_api.compare(a, worse, 0.03) == 1


@requires_loopback
def test_bench_run_level(bench, base_url):
    res = bench.run_level(base_url, ["how many employees?"], n=6, concurrency=2)
    assert res["ok"] == 6 and not res["errors"]
    assert res["rps"] > 0 and res["server_mean_ms"] == 650
    assert bench.percentile([1, 2, 3, 4, 5], 50) == 3
    assert bench.percentile([], 95) is None


def test_eval_set_schema(root):
    seen = set()
    for line in (root / "data/test_sets/eval_set_v2.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        row = json.loads(line)
        assert row["id"] not in seen
        seen.add(row["id"])
        assert row["question"].strip() and row.get("category")
        exp = row.get("expect", {})
        assert exp.get("refuse") or row.get("skip") or exp.get("sources")
    assert len(seen) >= 10


def test_purge_script_deletes_only_the_target_source(purge, tmp_path):
    chromadb = pytest.importorskip("chromadb")
    client = chromadb.PersistentClient(path=str(tmp_path))
    col = client.get_or_create_collection(purge.COLLECTION)
    col.add(
        ids=["a1", "a2", "b1"],
        embeddings=[[0.1, 0.2, 0.3], [0.2, 0.1, 0.3], [0.3, 0.2, 0.1]],
        metadatas=[
            {"company": "HDFC Bank", "source_file": "hdfcsec.pdf"},
            {"company": "HDFC Bank", "source_file": "hdfcsec.pdf"},
            {"company": "TCS", "source_file": "tcs.pdf"},
        ],
    )
    assert purge.list_sources(col, "hdfc") == {("HDFC Bank", "hdfcsec.pdf"): 2}
    assert purge.purge(col, "hdfcsec.pdf", apply=False) == (2, 3)  # dry run deletes nothing
    assert purge.purge(col, "hdfcsec.pdf", apply=True) == (2, 1)
    assert col.get()["ids"] == ["b1"]
