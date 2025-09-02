#!/usr/bin/env python3
"""
Testing script for Agentic Finance Framework
Adds 6 months of sample data and tests query capabilities
"""

import requests
import json
import random
from datetime import datetime, timedelta
from typing import List, Dict

# Configuration
BASE_URL = "http://127.0.0.1:8000"  # Update if different
USER_ID = "test_user_2025"

# Sample transaction data for Jan-June 2025
SAMPLE_TRANSACTIONS = [
    # January 2025
    "Received salary of ₹85000 on 1st January 2025",
    "Spent ₹1200 on groceries at BigBasket on 3rd January",
    "Paid ₹25000 rent on 5th January 2025",
    "Coffee expense of ₹350 on 7th January",
    "Received freelance payment ₹15000 on 10th January",
    "Electricity bill payment ₹2800 on 12th January",
    "Shopping expense ₹4500 at Amazon on 15th January",
    "Fuel expense ₹3200 on 18th January",
    "Restaurant bill ₹1800 on 20th January",
    "Mobile recharge ₹599 on 22nd January",
    
    # February 2025
    "Salary credited ₹85000 on 1st February 2025",
    "Grocery shopping ₹1400 on 3rd February",
    "Rent payment ₹25000 on 5th February 2025",
    "Medical expense ₹3500 on 8th February",
    "Dividend received ₹8000 on 10th February",
    "Internet bill ₹1299 on 12th February",
    "Clothing purchase ₹6700 on 14th February",
    "Petrol expense ₹3100 on 16th February",
    "Movie tickets ₹800 on 18th February",
    "Investment in mutual fund ₹20000 on 25th February",
    
    # March 2025
    "Monthly salary ₹85000 credited on 1st March 2025",
    "Vegetable shopping ₹800 on 2nd March",
    "House rent ₹25000 paid on 5th March",
    "Car maintenance ₹8500 on 8th March",
    "Bonus received ₹30000 on 10th March",
    "Gas cylinder ₹850 on 12th March",
    "Electronics purchase ₹15000 on 15th March",
    "Uber rides ₹1200 on 18th March",
    "Dining out ₹2200 on 20th March",
    "Gym membership ₹2500 on 25th March",
    
    # April 2025
    "Salary of ₹85000 received on 1st April 2025",
    "Monthly groceries ₹1600 on 4th April",
    "Rent payment ₹25000 on 5th April",
    "Travel booking ₹12000 on 8th April",
    "Consulting income ₹25000 on 12th April",
    "Water bill ₹450 on 15th April",
    "Book purchases ₹1800 on 18th April",
    "Taxi fare ₹650 on 20th April",
    "Pizza order ₹1200 on 22nd April",
    "Insurance premium ₹8000 on 28th April",
    
    # May 2025
    "Salary credited ₹85000 on 1st May 2025",
    "Grocery expenses ₹1300 on 3rd May",
    "Monthly rent ₹25000 on 5th May",
    "Medical checkup ₹5000 on 8th May",
    "Stock dividend ₹12000 on 10th May",
    "Broadband bill ₹1199 on 12th May",
    "Summer clothes ₹8500 on 15th May",
    "Flight tickets ₹18000 on 18th May",
    "Restaurant expense ₹2800 on 20th May",
    "SIP investment ₹15000 on 25th May",
    
    # June 2025
    "June salary ₹85000 on 1st June 2025",
    "Fresh groceries ₹1100 on 2nd June",
    "Rent paid ₹25000 on 5th June",
    "Laptop purchase ₹65000 on 8th June",
    "Project payment ₹40000 on 12th June",
    "Credit card bill ₹15000 on 15th June",
    "Vacation expenses ₹25000 on 18th June",
    "Metro card recharge ₹1000 on 20th June",
    "Coffee shop ₹450 on 22nd June",
    "Mutual fund SIP ₹20000 on 28th June"
]

# Test queries to ask after data ingestion
TEST_QUERIES = [
    "How much did I spend on rent in the first quarter of 2025?",
    "What were my total grocery expenses from January to March?",
    "Show me all my salary payments in 2025",
    "How much did I earn from freelance and consulting work?",
    "What were my biggest expenses in each month?",
    "Calculate my total income vs expenses for Q1 2025",
    "What did I spend on transportation (fuel, petrol, uber, taxi)?",
    "Show me all my investment transactions",
    "What were my dining and restaurant expenses?",
    "How much did I spend on shopping and electronics?",
    "What bills did I pay regularly each month?",
    "Show me my medical and healthcare expenses",
    "Calculate my monthly average spending",
    "What were my one-time big purchases over ₹10000?",
    "Show me my expense pattern by category"
]

class FinanceTester:
    def __init__(self, base_url: str, user_id: str):
        self.base_url = base_url
        self.user_id = user_id
        self.session = requests.Session()
    
    def add_transaction(self, text: str) -> Dict:
        """Add a single transaction"""
        url = f"{self.base_url}/transactions/add"
        payload = {
            "user_id": self.user_id,
            "text": text
        }
        
        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error adding transaction '{text}': {e}")
            return {"error": str(e)}
    
    def query_system(self, query: str, want_plot: bool = False) -> Dict:
        """Query the system"""
        url = f"{self.base_url}/query"
        payload = {
            "user_id": self.user_id,
            "query": query,
            "want_plot": want_plot
        }
        
        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error querying '{query}': {e}")
            return {"error": str(e)}
    
    def bulk_add_transactions(self, transactions: List[str]) -> List[Dict]:
        """Add multiple transactions"""
        results = []
        print(f"Adding {len(transactions)} transactions...")
        
        for i, transaction in enumerate(transactions, 1):
            print(f"Adding transaction {i}/{len(transactions)}: {transaction[:50]}...")
            result = self.add_transaction(transaction)
            results.append(result)
            
            # Small delay to avoid overwhelming the server
            import time
            time.sleep(0.1)
        
        return results
    
    def run_test_queries(self, queries: List[str]) -> List[Dict]:
        """Run test queries"""
        results = []
        print(f"\nRunning {len(queries)} test queries...")
        
        for i, query in enumerate(queries, 1):
            print(f"\nQuery {i}/{len(queries)}: {query}")
            result = self.query_system(query, want_plot=(i % 5 == 0))  # Plot every 5th query
            
            if "error" not in result:
                print(f"Answer: {result.get('answer_text', 'No answer')}")
                print(f"Citations: {len(result.get('citations', []))}")
                print(f"Verified Total: ₹{result.get('verified_total', 0)}")
                if result.get('plot_path'):
                    print(f"Plot: {self.base_url}{result['plot_path']}")
            else:
                print(f"Error: {result['error']}")
            
            results.append({"query": query, "result": result})
            
            # Delay between queries
            import time
            time.sleep(0.5)
        
        return results
    
    def test_server_health(self) -> bool:
        """Test if server is running"""
        try:
            response = self.session.get(f"{self.base_url}/docs")
            return response.status_code == 200
        except:
            return False

def main():
    print("=== Agentic Finance Framework Tester ===\n")
    
    # Initialize tester
    tester = FinanceTester(BASE_URL, USER_ID)
    
    # Check server health
    print("Checking server health...")
    if not tester.test_server_health():
        print(f"❌ Server not responding at {BASE_URL}")
        print("Please ensure your server is running with: uvicorn main:app --reload")
        return
    print("✅ Server is running\n")
    
    # Add sample transactions
    print("=" * 50)
    print("PHASE 1: Adding Sample Transactions")
    print("=" * 50)
    
    transaction_results = tester.bulk_add_transactions(SAMPLE_TRANSACTIONS)
    successful_adds = sum(1 for r in transaction_results if "error" not in r)
    print(f"Successfully added {successful_adds}/{len(SAMPLE_TRANSACTIONS)} transactions\n")
    
    # Run test queries
    print("=" * 50)
    print("PHASE 2: Testing Query System")
    print("=" * 50)
    
    query_results = tester.run_test_queries(TEST_QUERIES)
    
    # Summary
    print("\n" + "=" * 50)
    print("TESTING SUMMARY")
    print("=" * 50)
    print(f"Transactions added: {successful_adds}/{len(SAMPLE_TRANSACTIONS)}")
    print(f"Queries tested: {len(query_results)}")
    
    successful_queries = sum(1 for r in query_results if "error" not in r["result"])
    print(f"Successful queries: {successful_queries}/{len(query_results)}")
    
    # Save detailed results
    with open(f"test_results_{USER_ID}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        json.dump({
            "user_id": USER_ID,
            "transactions": transaction_results,
            "queries": query_results,
            "summary": {
                "total_transactions": len(SAMPLE_TRANSACTIONS),
                "successful_transactions": successful_adds,
                "total_queries": len(query_results),
                "successful_queries": successful_queries
            }
        }, f, indent=2)
    
    print(f"\nDetailed results saved to test_results_{USER_ID}_*.json")
    
    # Additional manual testing suggestions
    print("\n" + "=" * 50)
    print("MANUAL TESTING SUGGESTIONS")
    print("=" * 50)
    print(f"1. Open Swagger UI: {BASE_URL}/docs")
    print("2. Test individual endpoints:")
    print("   - POST /transactions/add")
    print("   - POST /query")
    print("   - POST /feedback")
    print("3. Try these additional queries:")
    print("   - 'What's my highest expense category?'")
    print("   - 'Show me April 2025 summary with plot'")
    print("   - 'Compare my income vs expenses'")
    print("   - 'Find all transactions over 20000 rupees'")

if __name__ == "__main__":
    main()