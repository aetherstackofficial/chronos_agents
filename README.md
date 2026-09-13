# Project Chronos: Distributed Market Digital Twin

Project Chronos is a high-performance, distributed stock market simulation powered by Agent-Based Modeling and Deep Reinforcement Learning. It generates a living, reactive Level 2 Limit Order Book (LOB) populated by concurrent AI and algorithmic agents, providing a realistic sandbox for financial backtesting and market shock analysis.

## 🚀 System Architecture

Chronos abandons traditional static backtesting in favor of a distributed microservice architecture, now fully containerized via Docker. It operates via a pull-based **"Notice Board" Protocol** over TCP/IP, allowing independent agent swarms to run asynchronously across multiple Docker containers without suffering from OS-level CPU starvation or thread blocking.

### Core Microservices
1. **The True Matching Engine:** An ultra-fast, native Python Limit Order Book that actively matches Bids and Asks based on price priority, manages partial fills, and generates real-time market state payloads.
2. **The Agent Swarm:** 100 concurrent algorithmic agents (Market Makers, Whales, Retail) communicating via brokerless Inter-Process Communication (IPC) over ZeroMQ.
3. **The FastAPI Bridge:** A high-speed asynchronous middleware layer that pulls state from the Engine at 20 frames-per-second and broadcasts it to the frontend via WebSockets.
4. **The React UI Dashboard:** A stunning, interactive Vite/React frontend utilizing `lightweight-charts` to visualize real-time candlestick data, Order Book depth, Live Tape, and an Agent PnL Leaderboard.
5. **The Macro Oracle:** An autonomous sentiment engine powered by the Google Gemini API (with a robust local heuristic fallback) that triggers exogenous market shocks based on injected breaking financial news.

## 🛠️ Tech Stack

* **Frontend:** React, Vite, Vanilla CSS, Lightweight-Charts
* **Backend & API:** Python 3.11, FastAPI, WebSockets
* **Networking Layer:** ZeroMQ (`pyzmq`) over TCP
* **Machine Learning:** PyTorch, Stable-Baselines3 (PPO), OpenAI Gymnasium, Google GenAI SDK
* **Infrastructure:** Docker, Docker Compose, Nginx

## 🧠 The Agent Ecosystem

The simulation is driven by three distinct classes of market participants, utilizing a **Shared Brain Pattern** to optimize RAM utilization across the distributed network. 

* **Market Makers (15 Agents):** High-Frequency traders initialized with $1M–$5M capital. They dynamically adjust their Bid/Ask spreads based on inventory risk and Order Flow toxicity.
* **Institutional Whales (4 Agents):** Smart money initialized with $5M–$20M capital. Trained via PPO to ride macroeconomic trends and execute massive block trades while utilizing a Hysteresis deadzone to minimize commission burn.
* **The Retail Swarm (80 Agents):** Heuristic algorithms initialized with $10k–$100k capital simulating chaotic, emotional trading based on real-time RSI crossovers. Controlled by a **Liquidation Engine** that actively monitors cash balances, instantly liquidating and respawning bankrupt agents to guarantee continuous market liquidity.

## 🌪️ Market Physics & Dynamics

* **Time Physics & Pausing:** The Master Engine utilizes a deterministic physics loop that ticks exactly 1 simulated minute per second. The UI can instantly freeze the Master Clock, dynamically pausing the entire distributed Swarm safely.
* **Baseline Brownian Noise:** The engine continuously injects randomized synthetic volume into the Order Flow Imbalance (OFI). This noise scales dynamically with overall market volume, perfectly mimicking the microscopic algorithmic jitter of a tier-1 exchange.
* **Macro-Injections & Fallback Matrix:** The Gemini Oracle parses breaking news sentiment (-1.0 to +1.0) injected from the UI and slams the engine with massive liquidity sweeps on severe events. A robust local Fallback Matrix ensures 100% uptime even if the LLM API hits rate limits or internet connectivity drops.

## 📦 Installation & Setup

Chronos is entirely containerized. You no longer need to manually manage Python virtual environments, Node modules, or complex network ports.

**1. Clone the repository:**
```bash
git clone https://github.com/aetherstackofficial/chronos_agents.git
cd chronos_agents
```

**2. Configure Environment Variables (Optional):**

Create a .env file in the root directory to supply your Google Gemini API key for the Oracle. If left blank, the Oracle will automatically use the Local Heuristic Fallback Engine.
```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

**3. Build and Launch the Distributed Network:**
```bash
docker-compose up --build
```
Docker will automatically build the React frontend, setup Nginx, download the Python ML environments, and orchestrate all 5 microservices in the correct dependency order.

## 🖥️ Running the Simulation

Once the Docker containers are running, the entire Chronos ecosystem is online.

1. Open your browser and navigate to: http://localhost:3000
2. Input your desired Asset ticker and Starting Price in the Header.
3. Click START SIMULATION to ignite the Master Engine and unleash the Swarm.
4. Type breaking financial news into the Injector input and click INJECT NEWS to watch the Oracle crash or pump the market in real-time.

Built for advanced distributed systems research, ML architectural design, and algorithmic trading simulations.
