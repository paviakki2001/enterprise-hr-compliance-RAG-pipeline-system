# HR & Compliance RAG - Week 1

## Objective
Build the data engineering foundation for a production-ready RAG system.

## What this notebook does
- Creates the required repository structure
- Loads datasets programmatically using Hugging Face
- Converts rows into document-style `.txt` files
- Enriches each document with metadata
- Chunks text using token-aware chunking
- Validates chunk quality and metadata completeness
- Saves processed outputs for Week 2 retrieval work

## Metadata Schema
Each document/chunk includes:
- department
- document_type
- category
- region
- year
- source
- file_name

## Chunking Strategy
- Chunk size: 900 tokens
- Overlap: 180 tokens

## Validation Checks
- Empty chunks
- Duplicate chunks
- Average chunk length
- Metadata completeness

## Known Limitations
- Public datasets are used as proxies for HR/compliance-style corpora
- Exact dataset availability may vary over time on Hugging Face
- Week 1 focuses on pipeline readiness, not embeddings or LLMs
