import threading
import asyncio
import time
import logging
import sys
from agents.live_agents import LiveMarketMaker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [THREAD-%(threadName)s] - %(message)s')

def agent_worker(agent_id):
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    agent = LiveMarketMaker(agent_id)
    
    try:
        loop.run_until_complete(agent.run())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()

if __name__ == "__main__":
    print("📈 Booting HFT Market Makers in isolated threads...")
    threads = []
    
    # Spawn 15 separate OS threads
    for i in range(15):
        t = threading.Thread(target=agent_worker, args=(f"MM_{i}",), name=f"MM_{i}", daemon=True)
        threads.append(t)
        t.start()
        time.sleep(0.2) 
        
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down all Market Maker threads...")