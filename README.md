# Project Chronos: Distributed Market Digital Twin

Project Chronos is a high-performance, distributed stock market simulation powered by Agent-Based Modeling and Deep Reinforcement Learning. It generates a living, reactive Level 2 Limit Order Book (LOB) populated by concurrent AI and algorithmic agents, providing a realistic sandbox for financial backtesting and market shock analysis.

## 🚀 System Architecture

Chronos abandons traditional static backtesting in favor of a distributed microservice architecture. It operates via a pull-based **"Notice Board" Protocol** over TCP/IP, allowing independent agent swarms to run asynchronously across multiple physical or virtual nodes without suffering from OS-level CPU starvation or thread blocking.

### Core Components
1. **The True Matching Engine (Server):** An ultra-fast, native Python dictionary-based Limit Order Book that actively matches Bids and Asks based on price priority, manages partial fills, and generates real-time market state JSON payloads (VWAP, 14-period RSI, Volatility, Order Flow Imbalance).
2. **The Agent Swarm (Clients):** 100 concurrent algorithmic agents communicating via brokerless Inter-Process Communication (IPC).
3. **The AI Brains (PyTorch/PPO):** Stable-Baselines3 Reinforcement Learning models trained to act as High-Frequency Market Makers and Institutional Whales.
4. **The Macro Oracle (LLM Client):** An autonomous sentiment engine powered by the Google Gemini API that triggers exogenous market shocks based on simulated breaking financial news.

## 🛠️ Tech Stack

* **Core Engine:** Python 3.11+, `asyncio`
* **Networking Layer:** ZeroMQ (`pyzmq`) over TCP
* **Machine Learning:** PyTorch, Stable-Baselines3 (PPO), OpenAI Gymnasium, Google GenAI SDK
* **Data Processing:** Pandas, Numpy
* **Architecture:** Microservices, Event-Driven, Distributed Systems

## 🧠 The Agent Ecosystem

The simulation is driven by three distinct classes of market participants, utilizing a **Shared Brain Pattern** to optimize RAM utilization across the distributed network. 

* **Market Makers (15 Agents):** High-Frequency traders initialized with $1M–$5M capital. They dynamically adjust their Bid/Ask spreads based on inventory risk and Order Flow toxicity.
* **Institutional Whales (4 Agents):** Smart money initialized with $5M–$20M capital. Trained via PPO to ride macroeconomic trends and execute massive block trades while utilizing a Hysteresis deadzone to minimize commission burn.
* **The Retail Swarm (80 Agents):** Heuristic algorithms initialized with $10k–$100k capital simulating chaotic, emotional trading based on real-time RSI crossovers. Controlled by a **Liquidation Engine** that actively monitors cash balances, instantly liquidating and respawning bankrupt agents to guarantee continuous market liquidity.

## 🌪️ Market Physics & Dynamics

* **Dynamic Participation:** Agents operate on randomized sleep/wake cycles. During quiet periods, only 30% of agents actively query the book to reduce network strain. Major news events awaken 100% of the swarm.
* **Baseline Brownian Noise:** The engine continuously injects randomized synthetic volume into the Order Flow Imbalance (OFI). This noise scales dynamically with overall market volume, perfectly mimicking the microscopic algorithmic jitter of a tier-1 exchange.
* **Macro-Injections & Fallback Matrix:** The Gemini Oracle parses breaking news sentiment (-1.0 to +1.0) and slams the engine with massive liquidity sweeps on severe events. A robust local Fallback Matrix ensures 100% uptime even if the LLM API hits rate limits.

## ⚙️ Concurrency Features

* **Pull-Based Notice Board:** Replaced legacy PUB/SUB broadcasts with a strict REQ/REP architecture, drastically reducing network bandwidth and CPU overhead.
* **Deadlock Protection:** The Engine utilizes a `zmq.Poller` with strict 50ms timeouts to prevent system hangs if an agent node crashes mid-execution.
* **High-Throughput IPC:** Capable of processing over 1,200 complex Neural Network forward passes and Level 2 JSON payload serializations per second locally.

## 📦 Installation & Setup

**1. Clone the repository:**
```bash
git clone [https://github.com/aetherstackofficial/chronos_agents.git](https://github.com/aetherstackofficial/chronos_agents.git)
cd chronos-agents

2. **Create and activate a virtual environment:**

   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate

3. **Install Dependencies:**

   # Navigate to the orchestrator directory where the requirements file lives
   cd swarm_orchestrator

   python -m pip install -r requirements.txt
   # OR install manually:
   python -m pip install pyzmq stable-baselines3[extra] gymnasium pandas numpy python-dotenv google-genai

4. **Configure Environment Variables:**

   ZMQ_HOST=127.0.0.1
   ZMQ_ORDER_PORT=5555
   GEMINI_API_KEY=your_gemini_api_key_here
```

## 🖥️ Running the Simulation

Due to the distributed microservice architecture, the components must be started in separate terminal processes. Ensure your virtual environment is activated in all three terminals.

```
Step 1: Start the Central Engine
   Open Terminal 1 and run the Matching Engine:
   cd swarm_orchestrator
   python true_engine.py

   The Engine will bind to tcp://127.0.0.1:5555 and wait for the swarm.

Step 2: Start the AI Swarm
   Open Terminal 2 and initialize the agents:
   cd swarm_orchestrator
   python main.py

   The Swarm loads the PyTorch models into RAM, connects to the Engine, and begins pulling data and executing trades.

Step 3: Start the Macro Oracle
   Open Terminal 3:
   cd swarm_orchestrator
   python oracle.py

   The Oracle will monitor news feeds and inject sentiment-driven market shocks into the ecosystem.
```
# Built for advanced distributed systems research and algorithmic trading simulations.
   
   
