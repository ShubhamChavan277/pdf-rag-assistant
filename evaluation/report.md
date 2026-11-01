# PDF RAG Assistant — Evaluation Report

## 1. Objective

Evaluate PDF parsing, chunking, semantic retrieval, reranking, and evidence-grounded answer generation using a small local benchmark.

## 2. Evaluation Dataset

- Source documents: 2 local PDF files.
- Total pages: 7.
- Total questions: 20.
- Answerable questions: 16.
- Unanswerable questions: 4.
- Retrieval evaluation: all 16 answerable questions.
- Generation evaluation: 2 answerable and 4 unanswerable questions.

## 3. Parser Comparison

Both pypdf and pdfplumber extracted all seven pages without empty pages.

### Parser measurements

| Document | Parser | Pages | Characters | Time (seconds) |
|---|---|---:|---:|---:|
| IRCTC Cancellation Rules | pypdf | 4 | 8,554 | 0.110114 |
| IRCTC Cancellation Rules | pdfplumber | 4 | 8,290 | 0.322348 |
| Traffic Rules and Regulations | pypdf | 3 | 7,860 | 0.080120 |
| Traffic Rules and Regulations | pdfplumber | 3 | 7,751 | 0.209562 |

### Normalized page-text similarity

| Document | Page 1 | Page 2 | Page 3 | Page 4 | Mean |
|---|---:|---:|---:|---:|---:|
| IRCTC Cancellation Rules | 0.9982 | 0.9988 | 0.8265 | 0.3333 | 0.7892 |
| Traffic Rules and Regulations | 1.0000 | 1.0000 | 1.0000 | N/A | 1.0000 |

The lower IRCTC similarity is primarily attributable to differences in text ordering and layout, particularly on page 4. Inspection confirmed that substantive content was present in both parser outputs.

pypdf was faster in both observed measurements and remains the parser used for retrieval evaluation.

These are single-run measurements, not statistically robust parser performance comparisons.

## 4. Chunking and Indexing

Chunking configuration:

- Chunk size: 1,000 characters.
- Chunk overlap: 150 characters.
- Chunking performed separately for each PDF.
- Page provenance retained in chunk metadata.

### Chunk statistics

| Document | Pages | Chunks | Average Characters | Minimum | Maximum |
|---|---:|---:|---:|---:|---:|
| IRCTC Cancellation Rules | 4 | 11 | 846.64 | 477 | 988 |
| Traffic Rules and Regulations | 3 | 11 | 797.09 | 322 | 992 |
| **Total** | **7** | **22** | — | — | — |

Combined indexing:

- Total indexed chunks: 22.
- Indexing time: 1.632962 seconds.
- Vector store: in-memory ChromaDB.

## 5. Retrieval Methodology

- Retrieval candidates: 20.
- Evaluation cutoff: top 5.
- Baseline: semantic retrieval.
- Reranked retrieval: FlashRank passage reranking.
- Embedding model: sentence-transformers/all-MiniLM-L6-v2.
- Metrics: Hit@5, Recall@5, Precision@5, and Mean Reciprocal Rank (MRR).
- Aggregate metrics: macro averages over 16 answerable questions.

Metric definitions:

- **Hit@5:** Whether at least one relevant chunk appears in the top five.
- **Recall@5:** Coverage of gold relevant pages in the top-five results.
- **Precision@5:** Number of relevant chunks divided by five.
- **MRR:** Reciprocal rank of the first relevant chunk.

Precision is chunk-based, whereas recall is page-coverage-based. They therefore measure different aspects of retrieval quality.

## 6. Retrieval Results

| Metric | Baseline | Reranked |
|---|---:|---:|
| Hit@5 | 1.0000 | 1.0000 |
| Recall@5 | 1.0000 | 1.0000 |
| Precision@5 | 0.5250 | 0.5125 |
| MRR | 0.9500 | 1.0000 |

### Interpretation

Reranking improved MRR from 0.9500 to 1.0000.

The improvement came from question I08, where the relevant passage moved from rank 5 in baseline retrieval to rank 1 after reranking.

Hit@5 and Recall@5 were already 1.0000 for baseline retrieval and remained unchanged after reranking.

Precision@5 decreased slightly from 0.5250 to 0.5125.

Therefore, reranking improved ranking quality on this benchmark but did not demonstrate an across-the-board improvement across all retrieval metrics.

## 7. Unanswerable-Question Diagnostics

Four unanswerable questions were evaluated separately for retrieval diagnostics.

| Question ID | Retrieved Candidates | Diagnostic |
|---|---:|---|
| U01 | 20 | Recorded |
| U02 | 20 | Recorded |
| U03 | 20 | Recorded |
| U04 | 20 | Recorded |

Retrieval diagnostics alone do not establish whether the answer generator will abstain. Generation behavior is evaluated separately.

## 8. Generation Evaluation

Generation evaluation used the Groq-hosted `openai/gpt-oss-20b` model.

### Evaluation configuration

- Model: openai/gpt-oss-20b.
- Selected questions: 6.
- Answerable questions: 2.
- Unanswerable questions: 4.
- Planned API calls: 6.
- Successful generations: 6.
- Failed generations: 0.
- Configured maximum retries: 0.

### Generation outcomes

| Question ID | Type | Outcome | Assessment |
|---|---|---|---|
| I01 | Answerable | Cancellation charges correctly identified | Correct |
| T01 | Answerable | Left-side driving rule correctly identified | Correct |
| U01 | Unanswerable | Exact abstention message | Appropriate abstention |
| U02 | Unanswerable | Exact abstention message | Appropriate abstention |
| U03 | Unanswerable | Exact abstention message | Appropriate abstention |
| U04 | Unanswerable | Specific explanation that personal booking status was unavailable | Appropriate semantic abstention |

Both answerable responses were consistent with the supplied PDF evidence and included source references.

All four unanswerable questions received appropriate abstentions. Three used the exact standard abstention message; U04 used a more specific explanation.

**Manual abstention assessment: 4/4 appropriate abstentions (100%).**

No automated LLM judge was used. Generated answers and supporting sources were manually reviewed.

## 9. Performance Measurements

All timings below are from the final generation-enabled benchmark run.

### Parser performance

| Document | pypdf (seconds) | pdfplumber (seconds) |
|---|---:|---:|
| IRCTC Cancellation Rules | 0.110114 | 0.322348 |
| Traffic Rules and Regulations | 0.080120 | 0.209562 |

pypdf was faster for both documents in this run.

### Indexing performance

| Measurement | Value |
|---|---:|
| Total chunks indexed | 22 |
| Indexing time | 1.632962 seconds |

### Retrieval and reranking latency

| Operation | Count | Mean (ms) | Median (ms) | P95 (ms) | Minimum (ms) | Maximum (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Retrieval | 16 | 39.668 | 17.528 | 131.766 | 10.001 | 131.766 |
| Reranking | 16 | 363.831 | 334.416 | 733.206 | 321.629 | 733.206 |

First reranking call: 733.206 ms.

### Generation latency

| Measurement | Value |
|---|---:|
| Successful generations | 6 |
| Mean | 763.039 ms |
| Median | 772.533 ms |
| P95 | 828.937 ms |
| Minimum | 685.996 ms |
| Maximum | 828.937 ms |

Generation latency measures the model-generation operation, not the deliberate pauses between API requests.

### Performance interpretation

These measurements represent a single benchmark execution and should not be treated as statistically stable performance guarantees.

Model initialization, cache state, local machine load, network conditions, and provider-side processing can affect observed latency. The first reranking call may include warm-up overhead.

## 10. Limitations

- The evaluation dataset contains only two PDF documents.
- Retrieval metrics are based on 16 answerable questions.
- Generation evaluation covers only two answerable and four unanswerable questions.
- No automated LLM judge was used.
- Generated answers and citations require manual review.
- Exact abstention-message matching does not capture semantically equivalent abstentions.
- Parser timings and latency measurements are single-run observations.
- Unanswerable-question coverage is limited to four examples.
- The benchmark evaluates the supplied PDF content, not the current legal validity of the information contained in those documents.
- Retrieval metrics do not directly measure final answer correctness or citation faithfulness.

## 11. Reproduction

Place the two local source PDFs in:

`evaluation/documents/`

These PDFs are intentionally excluded from version control.

Run retrieval evaluation:

```bash
python evaluation/run_evaluation.py
```

Run the optional six-question generation evaluation:

```bash
python evaluation/run_evaluation.py --generate
```

Raw timestamped JSON results are saved under:

`evaluation/results/`

The evaluation dataset, runner, metrics implementation, tests, report, and raw benchmark results are maintained in the project repository.
