# MovieMatch LLM evaluation handoff

**Workstream owners: Shlok and Krushik**  
**Date:** November 18, 2026  
## Implementation

ALS and Item-KNN merge their picks into a shortlist of 50 movies. Gemma 3 4B runs locally through Ollama at temperature 0 and sorts the shortlist into a top 10. The app checks that the list contains 10 distinct candidate IDs for a full shortlist. One correction request is allowed for an invalid response. If that response also fails, the app shows Stage 1's order with a fallback label.

The app writes explanations from MovieLens genres, the user's rating profile, and Item-KNN evidence. The earlier version let the model write explanations in free text. The app still uses the LLM for ranking and request interpretation.

The backend suite passed 50 tests and skipped two optional PyTorch tests. The frontend production build passed. In a separate live API check with MovieLens user 42 and a science-fiction request, the app returned ten distinct shortlist movies with explanations on the first attempt.

## Held-out evaluation protocol

- Trained ALS and Item-KNN on ratings through 2021. Evaluated on MovieLens ratings from 2022 through the dataset's end in 2023.
- Used a fixed 20-user sample with seed 42, including four users with ten or fewer earlier ratings. Scored all four methods on the same users. A held-out rating of 4 or higher counted as relevant.
- ALS and Item-KNN each had access to their full unseen candidate catalogs. The LLM and Stage 1 used exactly the same shortlist. The LLM received earlier ratings and tags only. Any LLM fallback counted as the original Stage 1 order.
- The comparison measures how well each method ranks movies users later rated highly. MovieLens has no text requests, plot summaries, or cast details in this setup. This run cannot measure how well the model follows a request such as “thoughtful sci-fi.”
- The 20 users are a small sample that intentionally includes cold-start users. NDCG@10 rewards placing relevant movies near the top. Recall@10 measures the share of relevant movies found in the first ten picks.

Reproduce the test with `python -m moviematch.cli train --with-validation --only als knn`, then `python -m moviematch.cli llm-pilot --split test --users 20` from the backend directory. Start Ollama with `OLLAMA_CONTEXT_LENGTH=8192 ollama serve` and follow the local model setup in the README. The command writes `artifacts/llm_pilot/test.json` under the configured artifacts directory.

## Held-out results

| Method | NDCG@10 | Recall@10 |
| --- | ---: | ---: |
| ALS | 0.1031 | 0.0412 |
| Item-KNN (sum) | 0.1202 | 0.0545 |
| Stage 1 (ALS + Item-KNN) | 0.1143 | 0.0629 |
| Stage 1 + LLM, including fallback | 0.1143 | 0.0629 |

- **Output reliability.** The LLM produced valid lists for 19/20 users, including 15 on the first try. The remaining 1 used the Stage 1 fallback. All 20 displayed lists contained ten distinct IDs from their respective shortlists.
- **Ranking.** Every user's NDCG@10 was identical to their Stage 1 score. The LLM changed the order or membership of the top ten for eight users, including three whose top ten contained different movies. Those changes left the ranking scores tied on this sample.
- **Explanation evidence.** An automated check matched all 190 explanations from accepted LLM lists to the exported genres, rating profiles, or Item-KNN evidence. The check measured factual support. It collected no human usefulness ratings.
- **Latency.** The median LLM time per user was 39.0 seconds on the local machine, including correction calls.

### Text for the presentation

**LLM reranking and evaluation — Shlok and Krushik**

Gemma 3 4B reranked 50 movies shortlisted by ALS and Item-KNN. On 20 held-out MovieLens users, it produced 19 valid top-10 lists and used Stage 1 for the other 1. NDCG@10 was 0.1143 for the LLM pipeline and 0.1143 for Stage 1. The app explains each pick using MovieLens and rating evidence.

## Earlier validation

- MovieLens 32M, with ratings before 2020 used for training and 2020–2021 ratings used for validation.
- Popularity, Item-KNN, ALS, and the merged Stage 1 shortlist were evaluated on 1,000 sampled users. Another 112 cold-start users were included for the cold-start group.
- A separate, fixed five-user pilot compared Stage 1, ALS, Item-KNN (sum scoring), and the local `gemma3:4b` LLM on the **same five users**. One had only two earlier ratings.
- The LLM received 50 Stage 1 candidates per user. User preferences and tags in this pilot were limited to information available before validation. No natural-language request was supplied, since held-out ratings do not contain one.

### Validation results

| Method | NDCG@10, 1,000 users | NDCG@10, same 5 pilot users | Recall@10, same 5 users |
| --- | ---: | ---: | ---: |
| ALS | 0.1175 | 0.3079 | 0.1814 |
| Item-KNN (sum) | 0.1226 | 0.3029 | 0.1368 |
| Stage 1 (merged) | 0.1208 | 0.3006 | 0.1704 |
| LLM or Stage 1 fallback | Not run on 1,000 users | 0.2948 | 0.1704 |

The five-user sample is too small for a reliable performance claim. Its scores must not be compared with the 1,000-user scores as if they used the same users. The LLM returned valid top-10 lists for **3 of 5** pilot users; **2 of 5** used the labeled Stage 1 fallback after invalid LLM responses. The accepted lists contained unique IDs from the 50 candidates. This pilot does **not** show an accuracy improvement over Stage 1 or ALS.

### Review of the original LLM-written explanations

The three accepted LLM responses contained 30 explanations. An assistant review flagged 15 with at least one detail or user-preference claim that was not supported by the fields sent to the LLM. Examples:

- *Crash* was described as having crime elements, but its supplied genres were Drama and Thriller and its supplied tags did not include crime.
- *Full Metal Jacket* was said to match the user's interest in War, but War was not among that user's supplied favorite genres.
- *Ocean's Eleven* was called a comedy, but the supplied genres were Crime and Thriller.

Flagged movie IDs by user: 16390 → 318, 1193; 87232 → 1483, 1222, 7090, 741, 714, 1090, 919; 155435 → 6874, 4226, 54995, 5989, 4963, 58559. The check treated genre, tag, story, and stated-preference claims as unsupported when the supplied candidate or user profile did not contain evidence for them.

The assistant checked claims against the supplied metadata. This review had no human raters and applies only to the original free-text explanations.

The 1,000-user classical results are in `artifacts/evaluation/results.json`. The archived five-user run is in `artifacts/llm_pilot/validation.json` locally and used the previous prompt and explanation format.
