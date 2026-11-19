# MovieMatch LLM handoff

**Shlok and Krushik · November 18, 2026**

ALS and Item-KNN shortlist 50 movies. Gemma 3 4B via Ollama reranks them into a top 10. The app rejects duplicate or out-of-shortlist IDs, retries once, and falls back to Stage 1 if needed. It writes explanations from MovieLens genres and rating evidence.

## Held-out MovieLens results

All methods used the same 20 users, including four with at most ten earlier ratings. ALS and Item-KNN were fitted on data through 2021. Held-out ratings of 4 or higher in 2022–2023 counted as relevant.

| Method | NDCG@10 | Recall@10 |
| --- | ---: | ---: |
| ALS | 0.1031 | 0.0412 |
| Item-KNN (sum) | 0.1202 | 0.0545 |
| Stage 1 (ALS + Item-KNN) | 0.1143 | 0.0629 |
| Stage 1 + LLM, including fallback | 0.1143 | 0.0629 |

- **Reliability.** Ollama returned 19 valid lists, including 15 on the first try. One user got the Stage 1 fallback.
- **Ranking.** The LLM changed the top ten for eight users and tied Stage 1 on NDCG@10 and Recall@10.
- **Explanations.** Automated checks matched all 190 explanations from accepted LLM lists to their source evidence. User satisfaction and explanation usefulness were outside this evaluation.
- **Demo.** A live science-fiction request returned ten valid IDs with explanations on the first try. Median Ollama time in the offline run was 39 seconds.

## Reproduce

Configure Ollama as in the [README](README.md), start it with `OLLAMA_CONTEXT_LENGTH=8192 ollama serve`, then run these commands from `backend`.

```bash
python -m moviematch.cli train --with-validation --only als knn
python -m moviematch.cli llm-pilot --split test --users 20
```

The output is `artifacts/llm_pilot/test.json` under the configured artifacts directory. The backend suite passed 50 tests and skipped two optional PyTorch tests. The frontend build passed.
