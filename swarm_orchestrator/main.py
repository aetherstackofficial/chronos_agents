# --- CRITICAL WINDOWS FIX: IMPORT AI/TORCH FIRST ---
from agents.live_agents import LiveWhaleAgent, LiveMarketMaker, LiveRetailAgent

# --- IMPORT NETWORKING SECOND ---
import asyncio
import sys
import logging
from core.loop_manager import SwarmOrchestrator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [SWARM] - %(message)s')

async def main():
    orchestrator = SwarmOrchestrator()
    swarm = []
    
    # Instantiate 15 PPO Market Makers
    for i in range(15):
        swarm.append(LiveMarketMaker(f"MM_{i}"))
        
    # Instantiate 4 PPO Hedge Funds (Whales)
    for i in range(4):
        swarm.append(LiveWhaleAgent(f"WHALE_{i}"))
        
    # Instantiate 80 Rule-Based Retail Traders
    for i in range(80):
        swarm.append(LiveRetailAgent(f"RETAIL_{i}"))
        
    orchestrator.load_agents(swarm)
    
    try:
        await orchestrator.run()
    except asyncio.CancelledError:
        pass
    finally:
        logging.info("Shutting down AI Swarm orchestrator...")
        orchestrator.shutdown()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())