import os
import zmq
import json
import logging
from google import genai
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [ORACLE] - %(message)s')

client = genai.Client()

def score_news_sentiment(headline, stock_name, current_price):
    """
    Forces Gemini to infer the sector and calculate a strict percentage impact.
    """
    prompt = f"""
    You are an expert quantitative analyst and risk manager.
    Evaluate the impact of the following breaking news on a specific publicly traded company.

    News Headline: "{headline}"
    Target Stock: "{stock_name}"
    Current Stock Price: ${current_price}

    Instructions:
    1. Identify the primary industry sector of "{stock_name}".
    2. Determine if the news affects this specific company (either directly or via a macro sector shock).
    3. Assign an impact score between -1.0 and 1.0. 
       * 1.0 represents an extreme euphoric event causing a maximum +20% surge in stock price.
       * -1.0 represents a catastrophic event causing a maximum -20% crash in stock price.
       * 0.0 means the news is completely irrelevant to the stock.

    Respond ONLY with a valid JSON object. Do not include any markdown formatting.
    Format exactly like this:
    {{
        "inferred_sector": "<string>",
        "impact_score": <float between -1.0 and 1.0>,
        "relevance": <boolean>,
        "reasoning": "<1-2 sentences explaining why>"
    }}
    """

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        
        # Clean potential markdown from the response
        raw_text = response.text.strip().replace("```json", "").replace("```", "")
        data = json.loads(raw_text)
        
        score = float(data.get("impact_score", 0.0))
        is_relevant = data.get("relevance", False)
        sector = data.get("inferred_sector", "UNKNOWN")
        reasoning = data.get("reasoning", "No reasoning provided.")
        
        logging.info(f"Gemini Inferred Sector: {sector}")
        logging.info(f"Gemini Reasoning: {reasoning}")
        
        if not is_relevant:
            return 0.0
            
        return max(-1.0, min(1.0, score))

    except Exception as e:
        logging.warning(f"Gemini API Error or JSON Parsing Failed: {e}")
        return 0.0

def execute_oracle_trade(engine_socket, sentiment_score, current_price):
    """
    Converts the percentage impact into the exact share quantity needed 
    to force the True Engine's math to gap the price perfectly.
    """
    if abs(sentiment_score) >= 0.1:
        side = "BUY" if sentiment_score > 0 else "SELL"
        
        # 1. Calculate target price gap (Max 20% change based on sentiment)
        max_percentage = 0.20
        target_gap_dollars = current_price * (abs(sentiment_score) * max_percentage)
        
        # 2. Reverse engineer the Engine's math: gap = (qty / 100,000) * $2.00
        # Therefore: qty = (target_gap / $2.00) * 100,000
        required_qty = int((target_gap_dollars / 2.00) * 100000)
        
        # 3. Ensure it hits the 50,000 minimum threshold to trigger a Macro Shock in the engine
        required_qty = max(50001, required_qty)

        logging.warning(f"ORACLE SHOCK! Target Gap: ${target_gap_dollars:.2f} | Executing {side} Order for {required_qty:,} shares.")

        engine_socket.send_json({
            "agent_id": "GEMINI_ORACLE",
            "action": side,
            "qty": required_qty,
            "type": "MARKET"
        })

        reply = engine_socket.recv_json()
        logging.info(f"Engine Execution Result: {reply}")
    else:
        logging.info("News impact too weak or irrelevant. No intervention required.")


def run_oracle():
    context = zmq.Context()

    # Connection to the Matching Engine
    engine_host = os.getenv("ZMQ_HOST", "127.0.0.1")
    engine_port = os.getenv("ZMQ_ORDER_PORT", "5555")
    engine_socket = context.socket(zmq.REQ)
    engine_socket.connect(f"tcp://{engine_host}:{engine_port}")

    # Connection to the UI Bridge
    oracle_listen_port = "5557"
    ui_listener = context.socket(zmq.REP)
    ui_listener.bind(f"tcp://0.0.0.0:{oracle_listen_port}")

    logging.info(f"Gemini Oracle Online. Listening for UI news on port {oracle_listen_port}...")

    try:
        while True:
            # Wait for news headline from the Dashboard
            message = ui_listener.recv_json()
            headline = message.get("headline", "")
            
            logging.info("=" * 60)
            logging.info(f"MANUAL INJECTION RECEIVED: {headline}")
            
            # Step 1: Ask the Engine what stock is currently active
            engine_socket.send_json({"action": "FETCH_STATE"})
            state = engine_socket.recv_json()
            active_stock = state.get("asset_id", "TCS")
            current_price = state.get("current_price", 100.0)
            
            logging.info(f"Targeting Active Stock: {active_stock} @ ${current_price:.2f}")
            
            # Step 2: Score the news specifically against that stock
            sentiment = score_news_sentiment(headline, active_stock, current_price)
            logging.info(f"Final Calculated Sentiment Score: {sentiment}")
            
            # Step 3: Inject the mathematically precise trade quantity
            execute_oracle_trade(engine_socket, sentiment, current_price)
            
            # Reply to the UI Bridge
            ui_listener.send_json({"status": "processed", "score": sentiment})
            logging.info("=" * 60)

    except KeyboardInterrupt:
        logging.info("Oracle shutting down.")
    finally:
        engine_socket.close()
        ui_listener.close()
        context.term()

if __name__ == "__main__":
    run_oracle()