# Search Engine Application

This application implements a web-based search engine that combines TF-IDF and Elasticsearch's BM25 search methods with PageRank scores for improved result ranking.

## Project Structure

```
root/
├── app.py                     # Main Flask application
├── pagerank_results.parquet   # Pre-computed PageRank scores
├── src/
│   └── resource/
│       └── tfidf_indexer.pkl  # Cached TF-IDF index data
├── crawled/                   # Contains crawled webpage data
│   └── [hash].txt            # JSON files with webpage content
└── Frontend/
    └── home.html             # Search interface template
```

## Prerequisites

### Python Version
- Python 3.7 or higher

### Required Python Packages
```bash
pip install numpy pandas elasticsearch flask scikit-learn scipy pyarrow
```

### Elasticsearch Setup
1. Install Elasticsearch 8.x from [elastic.co](https://www.elastic.co/downloads/elasticsearch)
2. Configure Elasticsearch to run on localhost:9200
3. Create a certificate for HTTPS connection
4. Update the following credentials in `app.py`:
   - Username (default: "elastic")
   - Password
   - Certificate path

## Setup Instructions

1. Unzip the project folder to your desired location

2. Create a Python virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure Elasticsearch is running and accessible at https://localhost:9200

5. Verify the file structure:
   - Ensure `pagerank_results.parquet` exists in the root directory
   - Verify `crawled/` directory contains webpage data files
   - Check that `Frontend/home.html` exists

## Running the Application

1. Start Elasticsearch if not already running

2. From the project root directory, run:
   ```bash
   python app.py
   ```

3. Access the search interface at: http://localhost:5000

## API Usage

The application provides two search endpoints:

### TF-IDF with PageRank Search
```
GET /search_tfidf_pr?query=<search_term>&weight=<pagerank_weight>
```

### Elasticsearch BM25 with PageRank Search
```
GET /search_es_pr?query=<search_term>&weight=<pagerank_weight>
```

Parameters:
- `query`: Search term (required)
- `weight`: PageRank weight (0-100, default: 30)

Example:
```
http://localhost:5000/search_tfidf_pr?query=camt&weight=30
```

## Response Format

```json
{
    "status": "success",
    "total_hit": <number>,
    "results": [
        {
            "url": "<string>",
            "title": "<string>",
            "text": "<string>",
            "relevant_text": "<string>",
            "score": <float>
        },
        ...
    ],
    "elapse": <float>
}
```

## Troubleshooting

1. Elasticsearch Connection Issues:
   - Verify Elasticsearch is running: `curl https://localhost:9200`
   - Check credentials in `app.py`
   - Ensure certificate path is correct

2. Missing Files:
   - Verify all required files exist in the correct locations
   - Check file permissions

3. Import Errors:
   - Confirm all required packages are installed
   - Verify Python version compatibility

## Notes

- The application uses pre-computed indices and PageRank scores for efficiency
- Text preprocessing includes lowercase conversion, punctuation removal, and whitespace normalization
- Search results combine relevance scores with PageRank using the specified weight
- Response text is truncated to 200 characters for preview
