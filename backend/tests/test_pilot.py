import dataclasses
import json

from moviematch.config import LlmSettings
from moviematch.llm import pilot
from tests.conftest import FakeLlm, valid_response_for


def test_validation_pilot_uses_exported_candidates(prepared, monkeypatch):
    settings = dataclasses.replace(
        prepared,
        llm=LlmSettings(True, "http://llm.invalid/v1", "", "test-model", 5, True, 0.0, True),
    )

    class FakeClient(FakeLlm):
        def close(self):
            pass

    def compact_response(messages):
        output = json.loads(valid_response_for(10)(messages))
        for movie in output["rankedMovies"]:
            movie.pop("rank")
            movie.pop("explanation")
        return json.dumps(output)

    fake = FakeClient([compact_response for _ in range(5)])
    monkeypatch.setattr(pilot, "OpenAICompatibleClient", lambda _: fake)
    report = json.loads(pilot.run_validation_pilot(settings, users=5, log=lambda _: None).read_text())

    assert report["split"] == "validation" and report["users"] == 5
    assert report["llmSuccesses"] == 5 and report["llmFallbacks"] == 0
    assert set(report["metricMeans"]) == {"stage1", "als", "item_knn_sum", "llm_or_fallback"}
    assert all(len(record["top10"]["llmOrFallback"]) == 10 for record in report["records"])
    assert all(len(record["explanations"]) == 10 for record in report["records"])
