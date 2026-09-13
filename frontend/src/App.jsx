import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import Header from './components/Header';
import Chart from './components/Chart';
import OrderBook from './components/OrderBook';
import Leaderboard from './components/Leaderboard';
import Tape from './components/Tape';

function App() {
    const [isRunning, setIsRunning] = useState(false);
    const [tickData, setTickData] = useState(null);
    const [lastKnownPrice, setLastKnownPrice] = useState(0);
    const wsRef = useRef(null);

    useEffect(() => {
        // Connect to FastAPI websocket
        const ws = new WebSocket('ws://localhost:8000/ws');
        wsRef.current = ws;

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'sim_update') {
                setTickData(data);
                if (data.ohlc) {
                    setLastKnownPrice(data.ohlc.close);
                }
            }
        };

        ws.onclose = () => {
            console.log('WebSocket disconnected');
        };

        return () => {
            ws.close();
        };
    }, []);

    const handleToggleSim = (payload) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
        
        if (isRunning) {
            wsRef.current.send(JSON.stringify({ action: 'stop_sim' }));
            setIsRunning(false);
            setTickData(null);
        } else {
            wsRef.current.send(JSON.stringify({ action: 'start_sim', ...payload }));
            setIsRunning(true);
        }
    };

    const handleInjectNews = (payload) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
        wsRef.current.send(JSON.stringify({ action: 'inject_news', ...payload }));
    };

    return (
        <div className="app-container">
            <Header 
                isRunning={isRunning} 
                lastKnownPrice={lastKnownPrice} 
                tickData={tickData} 
                onToggleSim={handleToggleSim}
                onInjectNews={handleInjectNews}
            />
            <Chart data={tickData} />
            <OrderBook book={tickData?.book} />
            <Leaderboard agents={tickData?.agents} />
            <Tape trades={tickData?.trades} />
        </div>
    );
}

export default App;
