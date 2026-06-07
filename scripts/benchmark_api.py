"""
Simple benchmark script for the Knowledge Assistant API.
"""

import asyncio
import time
import httpx
import sys

BASE_URL = "http://localhost:8000"

async def benchmark():
    async with httpx.AsyncClient(timeout=60.0) as client:
        # 1. Register/Login to get a token
        email = f"bench-{int(time.time())}@test.com"
        password = "Password123!"
        
        print(f"Registering user {email}...")
        try:
            resp = await client.post(f"{BASE_URL}/auth/register", json={
                "email": email,
                "password": password,
                "full_name": "Benchmark User"
            })
            resp.raise_for_status()
        except Exception as e:
            print(f"Registration failed: {e}")
            return

        print("Logging in...")
        resp = await client.post(f"{BASE_URL}/auth/login", json={
            "email": email,
            "password": password
        })
        token = resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 2. Run concurrent queries
        queries = [
            "What is the efficiency of a typical monocrystalline solar panel?",
            "How does MPPT work in a solar inverter?",
            "What are the NFPA 855 requirements for BESS clearance?",
            "Explain the difference between latent and sensible heat.",
            "How does building orientation affect HVAC load?"
        ]
        
        print(f"Starting benchmark with {len(queries)} concurrent queries...")
        
        start_time = time.perf_counter()
        tasks = [
            client.post(f"{BASE_URL}/query", json={"query": q}, headers=headers)
            for q in queries
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        print(f"\nTotal time for {len(queries)} concurrent queries: {total_time:.2f} seconds")
        
        success_count = 0
        for i, res in enumerate(results):
            if isinstance(res, httpx.Response):
                if res.status_code == 200:
                    success_count += 1
                    print(f"Query {i+1} success: {res.json().get('retrieval_time_ms', 0):.1f}ms retrieval")
                else:
                    print(f"Query {i+1} failed: {res.status_code} - {res.text}")
            else:
                print(f"Query {i+1} error: {res}")
        
        print(f"\nSuccess rate: {success_count}/{len(queries)}")

if __name__ == "__main__":
    try:
        asyncio.run(benchmark())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
