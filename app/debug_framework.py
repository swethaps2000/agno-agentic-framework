#!/usr/bin/env python3
"""
Debug script to identify query pipeline issues
"""

import requests
import json
from pymongo import MongoClient
import chromadb

# Configuration
BASE_URL = "http://127.0.0.1:8000"
USER_ID = "test_user_2025"
MONGODB_URI = "mongodb://localhost:27017"
MONGODB_DB = "agentic_finance"

def check_mongodb_data():
    """Check if data was properly stored in MongoDB"""
    print("=== CHECKING MONGODB DATA ===")
    try:
        client = MongoClient(MONGODB_URI)
        db = client[MONGODB_DB]
        
        # Check collections
        collections = ['income', 'expense', 'transactions_raw']
        for coll_name in collections:
            coll = db[coll_name]
            count = coll.count_documents({'user_id': USER_ID})
            print(f"{coll_name}: {count} documents")
            
            if count > 0:
                # Show sample document
                sample = coll.find_one({'user_id': USER_ID})
                print(f"Sample {coll_name}:", json.dumps({
                    '_id': str(sample['_id']),
                    'user_id': sample.get('user_id'),
                    'amount': sample.get('amount'),
                    'type': sample.get('type'),
                    'category': sample.get('category'),
                    'notes': sample.get('notes', '')[:50] + "..."
                }, indent=2))
        
        return True
    except Exception as e:
        print(f"MongoDB Error: {e}")
        return False

def check_chromadb_data():
    """Check if data was stored in ChromaDB"""
    print("\n=== CHECKING CHROMADB DATA ===")
    try:
        client = chromadb.PersistentClient(path='./.chromadb')
        collection = client.get_collection("transactions")
        
        count = collection.count()
        print(f"ChromaDB transactions collection: {count} documents")
        
        if count > 0:
            # Get a sample
            results = collection.get(limit=3)
            print("Sample documents:")
            for i, (doc_id, metadata, document) in enumerate(zip(
                results['ids'], results['metadatas'], results['documents']
            )):
                print(f"  {i+1}. ID: {doc_id}")
                print(f"     Metadata: {metadata}")
                print(f"     Document: {document[:100]}...")
        
        return True
    except Exception as e:
        print(f"ChromaDB Error: {e}")
        return False

def test_individual_components():
    """Test each component of the query pipeline individually"""
    print("\n=== TESTING INDIVIDUAL COMPONENTS ===")
    
    # Test 1: Import modules
    print("1. Testing imports...")
    try:
        from app.agents.retriever_agent import hybrid_retrieve
        from app.agents.reranker_agent import rerank
        from app.agents.fusion_agent import generate_answer
        print("✅ All imports successful")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test 2: Retrieval
    print("\n2. Testing retrieval...")
    try:
        candidates = hybrid_retrieve(USER_ID, "rent expenses")
        print(f"✅ Retrieved {len(candidates)} candidates")
        if candidates:
            print(f"Sample candidate: {candidates[0]}")
    except Exception as e:
        print(f"❌ Retrieval error: {e}")
        return False
    
    # Test 3: Reranking
    print("\n3. Testing reranking...")
    try:
        if candidates:
            top = rerank("rent expenses", candidates, top_k=3)
            print(f"✅ Reranked to {len(top)} results")
            if top:
                print(f"Top result score: {top[0].get('score', 'N/A')}")
    except Exception as e:
        print(f"❌ Reranking error: {e}")
        return False
    
    # Test 4: Answer generation
    print("\n4. Testing answer generation...")
    try:
        if 'top' in locals() and top:
            response = generate_answer(USER_ID, "rent expenses", top)
            print(f"✅ Generated answer: {response}")
    except Exception as e:
        print(f"❌ Answer generation error: {e}")
        return False
    
    return True

def test_minimal_query():
    """Test with a minimal manual query"""
    print("\n=== TESTING MINIMAL QUERY ===")
    
    try:
        payload = {
            "user_id": USER_ID,
            "query": "test",
            "want_plot": False
        }
        
        response = requests.post(f"{BASE_URL}/query", json=payload)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 500:
            try:
                error_detail = response.json()
                print(f"Error Detail: {error_detail}")
            except:
                print(f"Error Text: {response.text}")
        else:
            print(f"Response: {response.json()}")
            
    except Exception as e:
        print(f"Request error: {e}")

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("\n=== CHECKING DEPENDENCIES ===")
    
    required_packages = [
        'sentence_transformers',
        'chromadb', 
        'pymongo',
        'rank_bm25',
        'nltk',
        'pandas',
        'matplotlib'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\nInstall missing packages with:")
        print(f"pip install {' '.join(missing)}")
    
    return len(missing) == 0

def check_nltk_data():
    """Check and download NLTK data if needed"""
    print("\n=== CHECKING NLTK DATA ===")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        from nltk.tokenize import word_tokenize
        # Test tokenization
        tokens = word_tokenize("test sentence")
        print(f"✅ NLTK working: {tokens}")
        return True
    except Exception as e:
        print(f"❌ NLTK error: {e}")
        return False

def main():
    print("=== DEBUGGING FINANCE FRAMEWORK QUERY ISSUES ===\n")
    
    # Check each component
    steps = [
        ("Dependencies", check_dependencies),
        ("NLTK Data", check_nltk_data),
        ("MongoDB Data", check_mongodb_data), 
        ("ChromaDB Data", check_chromadb_data),
        ("Individual Components", test_individual_components),
        ("Minimal Query", test_minimal_query)
    ]
    
    results = {}
    for step_name, step_func in steps:
        print(f"\n{'='*50}")
        print(f"STEP: {step_name}")
        print('='*50)
        
        try:
            results[step_name] = step_func()
        except Exception as e:
            print(f"❌ {step_name} failed with exception: {e}")
            results[step_name] = False
    
    # Summary
    print(f"\n{'='*50}")
    print("DEBUGGING SUMMARY")
    print('='*50)
    
    for step_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{step_name}: {status}")
    
    # Recommendations
    print(f"\n{'='*50}")
    print("RECOMMENDATIONS")
    print('='*50)
    
    if not results.get("Dependencies", True):
        print("1. Install missing Python packages")
    
    if not results.get("NLTK Data", True):
        print("2. Fix NLTK installation and data")
        
    if not results.get("MongoDB Data", True):
        print("3. Check MongoDB connection and data")
        
    if not results.get("ChromaDB Data", True):
        print("4. Check ChromaDB setup and embeddings")
        
    if not results.get("Individual Components", True):
        print("5. Debug specific component failures")
        
    print("\n6. Check server logs for detailed error messages")
    print("7. Add error handling and logging to your endpoints")

if __name__ == "__main__":
    main()