# Graph-RAG Exam Answering Workflow

This project answers LEXam multiple‑choice questions with a custom **agentic graph‑RAG** pipeline implemented in `do_exam.py`.

## High‑level flow

1. **Issue extraction**
   - Send the exam question to the LLM.
   - The model must output **only** a Python list of strings named “questions juridiques traitées”.

2. **Issue embeddings**
   - Embed each extracted issue with the `gemma-3-300m-embedding` model (default: `together_ai/google/gemma-3-300m-embedding`).

3. **Graph retrieval (Neo4j)**
   - Query **DecisionEmbedding** nodes (one per vector) linked to **court_decisions**.
   - Vector search uses a Neo4j **vector index** with cosine similarity.
   - Aggregate best vector score per decision.
   - Run **BM25** over `questions juridiques traitees` using Neo4j full‑text index.
   - Combine vector similarity + BM25 and keep **top 10**.

4. **Document helpfulness scoring (parallel)**
   - Prompt the model **in parallel**, one prompt per court decision.
   - The model returns an **integer score (0–5)** for usefulness.
   - Keep only docs with **score ≥ 4**.

5. **Graph expansion (neighbors)**
   - For each high‑scoring doc, explore neighbors in the graph.
   - Filter neighbors by **cosine similarity** (close to the seed doc’s similarity).
   - Explore **up to 4 neighbors in parallel**, in order of closest cosine similarity.

6. **Stop by time budget**
   - A per‑question time budget stops exploration.

7. **Final answer**
   - Assemble all best documents into the context window.
   - Ask the LLM to produce the final exam answer.

## Script usage

```bash
python do_exam.py \
  --input_file data/MCQs_test_4.xlsx \
  --output_file outputs/answers.csv \
  --llm gpt-4o-mini
```

## Neo4j setup expectations

- **Decision nodes**: `court_decisions`
  - `questions juridiques traitees`: list of strings
- **Embedding nodes**: `DecisionEmbedding`
  - `embedding`: single vector
  - Relationship: `(court_decisions)-[:HAS_EMBEDDING]->(DecisionEmbedding)`
- **Vector index**:
  - Default name: `decision_embedding_idx`
  - Indexes `DecisionEmbedding.embedding` with cosine similarity
- **Full‑text index (BM25)**:
  - Default name: `court_decisions_qjt`
  - Indexes `questions juridiques traitees`

## Configuration (env vars / args)

- `NEO4J_URI` (default `bolt://localhost:7687`)
- `NEO4J_USER` (default `neo4j`)
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE` (default `neo4j`)
- `NEO4J_FULLTEXT_INDEX` (default `court_decisions_qjt`)
- `NEO4J_VECTOR_INDEX` (default `decision_embedding_idx`)

Key CLI args:

- `--embedding_model` (default `together_ai/google/gemma-3-300m-embedding`)
- `--top_k` (default `10`)
- `--vector_candidates_k` (default `200`)
- `--alpha` (vector/BM25 mix, default `0.6`)
- `--time_budget_s` (default `40`)
- `--neighbor_parallel` (default `4`)
- `--neighbor_min_sim` (default `0.6`)
- `--neighbor_max_delta` (default `0.15`)

## Output

A CSV with:
- `answer`: final model answer
- `used_doc_ids`: Neo4j node ids used in context
- `used_doc_scores`: helpfulness scores

## Notes

- BM25 is run through Neo4j full‑text index; adjust `--fulltext_index` if your index has a different name.
- Vector search uses the Neo4j vector index; adjust `--vector_index` if your index has a different name.
