# Evaluation report

Generated 2026-11-06T10:32:19+00:00 with `gemini` / `gemini-3.6-flash`.

## 1. Extraction accuracy

**33/33 = 100.0%** of ground-truth values matched (rules-based parser, no LLM).

## 2. Retrieval quality

Hit@3 on 12 hand-written questions (paraphrased, no key terms):

| Method | Hit rate |
|---|---|
| **Hybrid (BM25 + vector, RRF)** | **83.3%** |
| Vector only | 66.7% |
| BM25 only | 50.0% |

| Question | Quarter | Hybrid | Vector | BM25 |
|---|---|---|---|---|
| acquisition_margin | Q2 FY27 | yes | no | yes |
| holiday_shutdowns | Q2 FY27 | no | no | no |
| telecom_rampdown | Q2 FY27 | yes | yes | no |
| contractor_costs | Q2 FY27 | yes | yes | no |
| ai_model_partners | Q4 FY26 | no | no | no |
| manufacturing_deal | Q1 FY27 | yes | yes | yes |
| salary_headwind | Q4 FY26 | yes | yes | yes |
| ai_revenue_size | all | yes | yes | no |
| datacenter_business | Q2 FY27 | yes | yes | yes |
| campus_hiring | Q2 FY27 | yes | yes | no |
| order_book | Q1 FY27 | yes | no | yes |
| project_delays | all | yes | yes | yes |

## 3. Output faithfulness

**29/29 = 100.0%** of checks passed on `sample_output.json`.

- All checks passed.

## 4. Consistency

2 of 3 runs completed.

| Field | Most common | Agreement | Values |
|---|---|---|---|
| revenue_direction | growth | 50.0% | growth, flat |
| margin_direction | contraction | 100.0% | contraction, contraction |
| confidence | high | 50.0% | high, medium |
