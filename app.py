# Core data processing imports
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
import time
import pickle
import re
from scipy import sparse

# Web/API imports
from elasticsearch import Elasticsearch
from flask import Flask, request, render_template

# ML/Feature extraction imports
from sklearn.feature_extraction.text import TfidfVectorizer

# Parquet handling
import pyarrow.parquet as pq
import pyarrow as pa

class TFIDFIndexer:
    def __init__(self, is_reset=False):
        self.crawled_folder = Path(Path().absolute()) / "crawled/"
        self.stored_file = 'src/resource/tfidf_indexer.pkl'
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
        
        # Load the cached index if it exists and is_reset is False
        if not is_reset and os.path.isfile(self.stored_file):
            with open(self.stored_file, "rb") as f:
                cached_dict = pickle.load(f)
            self.__dict__.update(cached_dict)
        else:
            self.run_indexer()
            
    @staticmethod
    def preprocess_text(text):
        # Lowercase text
        text = text.lower()
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', '', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def run_indexer(self):
        documents = []

        # Load all .txt files from crawled folder
        for file in os.listdir(self.crawled_folder):
            # Skip files that aren't .txt or contain 'pdf' in the URL
            if not file.endswith(".txt"):
                continue
                
            file_path = os.path.join(self.crawled_folder, file)

            # Read the file with UTF-8 encoding and handle errors
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                try:
                    j = json.load(f)
                    # Skip entries with PDF URLs or missing required fields
                    if ('url' in j and '.pdf' in j['url'].lower()) or \
                    'title' not in j or 'text' not in j:
                        print(f"Skipped file {file}: PDF URL or missing required fields")
                        continue
                    
                    # Additional check for binary-looking content
                    if any(ord(c) < 32 and c not in '\n\r\t' for c in j.get('text', '')):
                        print(f"Skipped file {file}: Contains binary content")
                        continue
                    
                    # Add filename to the document
                    j['filename'] = file
                    
                    # Preprocess title and text
                    j['title'] = self.preprocess_text(j['title'])
                    j['text'] = self.preprocess_text(j['text'])
                    
                    # Skip if content is too short after preprocessing
                    if len(j['text']) < 50:  # Adjust minimum length as needed
                        print(f"Skipped file {file}: Content too short")
                        continue
                        
                    documents.append(j)
                    
                except:
                    print(f"Error reading {file}")

        if not documents:
            raise ValueError("No valid documents found. Ensure crawled/ folder contains valid .txt files.")

        # Create DataFrame and prepare corpus
        self.documents = pd.DataFrame.from_dict(documents)
        self.corpus = self.documents.apply(lambda s: ' '.join(s[['title', 'text']]), axis=1).tolist()

        if not self.corpus:
            raise ValueError("Corpus is empty. Ensure documents contain 'title' and 'text' fields.")

        # Fit TF-IDF vectorizer on corpus
        self.vectorizer.fit(self.corpus)
        self.tfidf_matrix = self.vectorizer.transform(self.corpus)

        # Ensure directory exists before saving
        os.makedirs(os.path.dirname(self.stored_file), exist_ok=True)

        # Save the final processed data
        with open(self.stored_file, "wb") as f:
            pickle.dump(self.__dict__, f)

    def search_query(self, query, top_n=5):
        # Preprocess query
        query = self.preprocess_text(query)
        
        # Transform query to TF-IDF space
        query_vector = self.vectorizer.transform([query])
        
        # Calculate TF-IDF similarity scores
        tfidf_scores = (self.tfidf_matrix @ query_vector.T).toarray().flatten()
        
        # Create results DataFrame
        results_df = self.documents.copy()
        results_df['score'] = tfidf_scores
        
        # Get top results
        top_results = results_df.nlargest(top_n, 'score')[
            ['url', 'title', 'text', 'filename', 'score']
        ]
        
        return top_results

def get_original_text(filename, url=None):
    """Get original text from crawled file based on filename or URL."""
    try:
        # First try with filename
        if pd.notna(filename):
            file_path = os.path.join('crawled', filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data['text']
        
        # If filename is NaN or file not found, try searching by URL
        if url:
            files = [f for f in os.listdir('crawled') if f.endswith('.txt')]
            for fname in files:
                try:
                    with open(os.path.join('crawled', fname), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if data['url'] == url or (url in data.get('url_lists', [])):
                            return data['text']
                except:
                    continue
        
        return ""
    except Exception as e:
        print(f"Error loading text from {filename} or {url}: {e}")
        return ""

def get_relevant_sentences(text, query, max_length=200):
    """Get Google-like snippet with query terms highlighted in context."""
    if not text or not query:
        return ""
    
    try:
        # Normalize text and query
        text = text.replace('\n', ' ').strip()
        query_terms = [term.lower() for term in query.split()]
        
        # Find the first occurrence of any query term
        text_lower = text.lower()
        positions = []
        
        # Find all occurrences of query terms
        for term in query_terms:
            pos = text_lower.find(term)
            if pos >= 0:
                positions.append(pos)
        
        if not positions:
            return text[:max_length] + "..." if len(text) > max_length else text
        
        # Get the first occurrence
        start_pos = min(positions)
        
        # Find sentence boundaries
        sentence_start = text.rfind('.', 0, start_pos)
        if sentence_start == -1:
            sentence_start = max(0, start_pos - 50)
        else:
            sentence_start += 1
            
        # Get enough context after the query term
        end_pos = start_pos + max_length
        sentence_end = text.find('.', end_pos)
        if sentence_end == -1:
            sentence_end = min(len(text), end_pos + 50)
        
        # Extract the relevant portion
        snippet = text[sentence_start:sentence_end].strip()
        
        # If snippet is too long, trim it while keeping the query term
        if len(snippet) > max_length:
            # Ensure we keep the part with the query term
            term_pos = min(positions) - sentence_start
            if term_pos < max_length/2:
                snippet = snippet[:max_length] + "..."
            else:
                start_offset = term_pos - int(max_length/3)
                snippet = "..." + snippet[start_offset:start_offset + max_length] + "..."
        
        return snippet
        
    except Exception as e:
        print(f"Error creating snippet: {e}")
        return text[:max_length] if text else ""

def process_tfidf_search(query_term, pr_weight=0.3):
    """Process TF-IDF search with PageRank combination but get text from ES."""
    start = time.time()
    response_object = {'status': 'success'}
    
    try:
        # Get initial TF-IDF results
        results_df = tfidf_indexer.search_query(query_term, top_n=100)
        
        # Get the nice formatted text from Elasticsearch for these URLs
        es_results = app.es_client.search(
            index='simple',
            source_excludes=['url_lists'],
            size=100,
            query={
                "terms": {
                    "url.keyword": results_df['url'].tolist()
                }
            }
        )
        
        # Create a mapping from URL to ES content
        es_content = {
            hit['_source']['url']: {
                'title': hit['_source']['title'],
                'text': hit['_source']['text'][:200],
            }
            for hit in es_results['hits']['hits']
        }
        
        # Update the text content while keeping TF-IDF scores
        results_df['title'] = results_df['url'].map(lambda x: es_content.get(x, {}).get('title', results_df.loc[results_df['url'] == x, 'title'].iloc[0]))
        results_df['text'] = results_df['url'].map(lambda x: es_content.get(x, {}).get('text', results_df.loc[results_df['url'] == x, 'text'].iloc[0]))
        
        # Get relevant text using the nice ES text
        results_df['relevant_text'] = results_df.apply(
            lambda row: get_relevant_sentences(
                get_original_text(row['filename']), 
                query_term
            ),
            axis=1
        )
        
        # Load PageRank scores
        pr_scores = pd.read_parquet('pagerank_results.parquet')
        
        # Normalize TF-IDF scores
        results_df['tfidf_norm'] = (results_df['score'] - results_df['score'].min()) / \
                                 (results_df['score'].max() - results_df['score'].min())
        
        # Add normalized PageRank scores
        results_df['pagerank'] = pr_scores.loc[results_df['url']]['score'].values
        results_df['pagerank_norm'] = (results_df['pagerank'] - results_df['pagerank'].min()) / \
                                    (results_df['pagerank'].max() - results_df['pagerank'].min())
        
        # Combine scores with weight
        results_df['final_score'] = (1 - pr_weight) * results_df['tfidf_norm'] + \
                                  pr_weight * results_df['pagerank_norm']
        
        # Resort by combined scores
        results_df = results_df.nlargest(100, 'final_score')
        
        # Prepare final output
        output_df = results_df[['url', 'title', 'text', 'relevant_text', 'final_score']]
        output_df = output_df.rename(columns={'final_score': 'score'})
        
        response_object['total_hit'] = len(output_df)
        response_object['results'] = output_df.to_dict('records')
        response_object['elapse'] = time.time() - start
        
        return response_object
        
    except Exception as e:
        print(f"Error in process_tfidf_search: {str(e)}")
        response_object['status'] = 'error'
        response_object['message'] = str(e)
        return response_object

def process_es_search(query_term, pr_weight=0.3):
    """Process Elasticsearch search with PageRank combination."""
    start = time.time()
    response_object = {'status': 'success'}
    
    try:
        # Get BM25 results from Elasticsearch
        results = app.es_client.search(
            index='simple',
            source_excludes=['url_lists'],
            size=100,
            query={
                "match": {
                    "text": query_term
                }
            }
        )
        
        # Convert to DataFrame
        results_df = pd.DataFrame([
            {
                'url': hit['_source']['url'],
                'title': hit['_source']['title'],
                'text': hit['_source']['text'][:200],
                'es_score': hit['_score']
            } 
            for hit in results['hits']['hits']
        ])
        
        # Get existing files from crawled folder
        crawled_files = {f for f in os.listdir('crawled') if f.endswith('.txt')}
        
        # Get filenames from TF-IDF indexer using URL as key
        url_to_filename = dict(zip(tfidf_indexer.documents['url'], tfidf_indexer.documents['filename']))
        results_df['filename'] = results_df['url'].map(url_to_filename)
        
        # Print debugging info for missing files
        missing_files = results_df[results_df['filename'].isna()]
        if not missing_files.empty:
            print("\nMissing files for URLs:")
            for url in missing_files['url']:
                print(f"No filename found for URL: {url}")
                
        mismatched_files = results_df[
            ~results_df['filename'].isna() & 
            ~results_df['filename'].isin(crawled_files)
        ]
        if not mismatched_files.empty:
            print("\nFiles not found in crawled folder:")
            for _, row in mismatched_files.iterrows():
                print(f"File {row['filename']} not found for URL: {row['url']}")
        
        # Get relevant text
        results_df['relevant_text'] = results_df.apply(
            lambda row: get_relevant_sentences(
                get_original_text(row['filename'], row['url']), 
                query_term
            ),
            axis=1
        )
        
        # Load PageRank scores
        pr_scores = pd.read_parquet('pagerank_results.parquet')
        
        # Normalize Elasticsearch scores
        results_df['es_norm'] = (results_df['es_score'] - results_df['es_score'].min()) / \
                               (results_df['es_score'].max() - results_df['es_score'].min())
        
        # Add normalized PageRank scores
        results_df['pagerank'] = pr_scores.loc[results_df['url']]['score'].values
        results_df['pagerank_norm'] = (results_df['pagerank'] - results_df['pagerank'].min()) / \
                                    (results_df['pagerank'].max() - results_df['pagerank'].min())
        
        # Combine scores with weight
        results_df['final_score'] = (1 - pr_weight) * results_df['es_norm'] + \
                                  pr_weight * results_df['pagerank_norm']
        
        # Resort by combined scores
        results_df = results_df.nlargest(100, 'final_score')
        
        # Prepare final output
        output_df = results_df[['url', 'title', 'text', 'relevant_text', 'final_score']]
        output_df = output_df.rename(columns={'final_score': 'score'})
        
        response_object['total_hit'] = len(output_df)
        response_object['results'] = output_df.to_dict('records')
        response_object['elapse'] = time.time() - start
        
        return response_object
        
    except Exception as e:
        print(f"Error in process_es_search: {str(e)}")
        response_object['status'] = 'error'
        response_object['message'] = str(e)
        return response_object

# Flask application setup
app = Flask(__name__, static_folder='Frontend', template_folder='Frontend')

# Initialize Elasticsearch client
app.es_client = Elasticsearch(
    "https://localhost:9200",
    basic_auth=("elastic", "+zu*TMbwCT-I9_fi3-L4"),
    ca_certs="~/http_ca.crt",
    verify_certs=True
)

# Initialize TF-IDF indexer
tfidf_indexer = TFIDFIndexer(is_reset=False)

# Routes
@app.route('/')
def home():
    return render_template("home.html")

@app.route('/search_tfidf_pr', methods=['GET'])
def search_tfidf_pr():
    # Get query parameters
    argList = request.args.to_dict(flat=False)
    query_term = argList['query'][0]
    
    # Convert weight from percentage to decimal (default 30%)
    weight_percent = float(argList.get('weight', [30])[0])
    pr_weight = weight_percent / 100
    
    # Return search results
    return process_tfidf_search(query_term, pr_weight)

@app.route('/search_es_pr', methods=['GET'])
def search_es_pr():
    # Get query parameters
    argList = request.args.to_dict(flat=False)
    query_term = argList['query'][0]
    
    # Convert weight from percentage to decimal (default 30%)
    weight_percent = float(argList.get('weight', [30])[0])
    pr_weight = weight_percent / 100
    
    # Return search results
    return process_es_search(query_term, pr_weight)

if __name__ == '__main__':
    app.run(debug=False)