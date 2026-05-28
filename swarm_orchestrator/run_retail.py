import threading
import asyncio
import time
import logging
import sys
from agents.live_agents import LiveRetailAgent

# Suppress standard INFO logs for Retail so it doesn't crash the terminal with text
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - [THREAD-%(threadName)s] - %(message)s')

def agent_worker(agent_id):
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    agent = LiveRetailAgent(agent_id)
    
    try:
        loop.run_until_complete(agent.run())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()

if __name__ == "__main__":
    print("🔥 Booting 80 Retail Traders in isolated threads... (This may take a moment)")
    threads = []
    
    # Spawn 80 separate OS threads
    for i in range(80):
        t = threading.Thread(target=agent_worker, args=(f"RETAIL_{i}",), name=f"RETAIL_{i}", daemon=True)
        threads.append(t)
        t.start()
        time.sleep(0.05) # Very fast stagger
        
    print("✅ All 80 Retail threads are live and fighting for execution!")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down all Retail threads...")