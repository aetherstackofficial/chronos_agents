import threading
import asyncio
import time
import logging
import sys
from agents.live_agents import LiveWhaleAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [THREAD-%(threadName)s] - %(message)s')

def agent_worker(agent_id):
    """This function runs inside a completely isolated OS thread."""
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    # 1. Create a brand new event loop for this specific thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # 2. Instantiate the agent INSIDE the thread (Ensures isolated ZMQ sockets)
    agent = LiveWhaleAgent(agent_id)
    
    # 3. Run the agent forever
    try:
        loop.run_until_complete(agent.run())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()

if __name__ == "__main__":
    print("🚀 Booting Institutional Whales in isolated threads...")
    threads = []
    
    # Spawn 4 separate OS threads
    for i in range(4):
        t = threading.Thread(target=agent_worker, args=(f"WHALE_{i}",), name=f"WHALE_{i}", daemon=True)
        threads.append(t)
        t.start()
        time.sleep(0.5) # Stagger connections to prevent network stampede
        
    try:
        # Keep the main process alive while daemon threads run
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down all Whale threads...")