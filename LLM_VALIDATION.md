# LLM validation update

**Workstream owners: Shlok and Krushik**  
**Date:** November 18, 2026  
**Status:** Exploratory validation, not the final test result

## What we ran

- MovieLens 32M, with ratings before 2020 used for training and 2020–2021 ratings used for validation.
- Popularity, Item-KNN, ALS, and the merged Stage 1 shortlist were evaluated on 1,000 sampled users. Another 112 cold-start users were included for the cold-start group.
- A separate, fixed five-user pilot compared Stage 1, ALS, Item-KNN (sum scoring), and the local `gemma3:4b` LLM on the **same five users**. One had only two earlier ratings.
- The LLM received 50 Stage 1 candidates per user. User preferences and tags in this pilot were limited to information available before validation. No natural-language request was supplied, since held-out ratings do not contain one.

## Results

| Method | NDCG@10, 1,000 users | NDCG@10, same 5 pilot users | Recall@10, same 5 users |
| --- | ---: | ---: | ---: |
| ALS | 0.1175 | 0.3079 | 0.1814 |
| Item-KNN (sum) | 0.1226 | 0.3029 | 0.1368 |
| Stage 1 (merged) | 0.1208 | 0.3006 | 0.1704 |
| LLM or Stage 1 fallback | Not run on 1,000 users | 0.2948 | 0.1704 |

The five-user sample is too small for a reliable performance claim. Its scores must not be compared with the 1,000-user scores as if they used the same users. The LLM returned valid top-10 lists for **3 of 5** pilot users; **2 of 5** used the labeled Stage 1 fallback after invalid LLM responses. The accepted lists contained unique IDs from the 50 candidates. This pilot does **not** show an accuracy improvement over Stage 1 or ALS.

## Explanation review

The three accepted LLM responses contained 30 explanations. An assistant review flagged 15 with at least one detail or user-preference claim that was not supported by the fields sent to the LLM. Examples:

- *Crash* was described as having crime elements, but its supplied genres were Drama and Thriller and its supplied tags did not include crime.
- *Full Metal Jacket* was said to match the user's interest in War, but War was not among that user's supplied favorite genres.
- *Ocean's Eleven* was called a comedy, but the supplied genres were Crime and Thriller.

Flagged movie IDs by user: 16390 → 318, 1193; 87232 → 1483, 1222, 7090, 741, 714, 1090, 919; 155435 → 6874, 4226, 54995, 5989, 4963, 58559. The check treated genre, tag, story, and stated-preference claims as unsupported when the supplied candidate or user profile did not contain evidence for them.

This is a strict metadata-grounding check by the assistant, **not a human rating study**. Shlok and Krushik should review a sample themselves before reporting a human faithfulness score. The app validates movie IDs and response structure; it does not yet guarantee that every explanation is factually supported.

## What to tell the team

The LLM is integrated with the combined ALS and Item-KNN shortlist. We tested it on real MovieLens validation users and compared it with the classical models. It produced valid recommendations for three of five pilot users, with safe fallbacks for two. The small pilot did not improve accuracy, and some explanations need work. Shlok and Krushik share the LLM evaluation workstream. The final test run and human explanation review are still pending.

Reproduce the local pilot with `cd backend && python -m moviematch.cli llm-pilot --users 5` after configuring Ollama. Its machine-readable output is `artifacts/llm_pilot/validation.json` and is intentionally not committed. The 1,000-user classical results are in `artifacts/evaluation/results.json`.
