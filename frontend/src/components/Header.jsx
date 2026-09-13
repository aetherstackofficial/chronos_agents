import React, { useState } from 'react';

const Header = ({ isRunning, lastKnownPrice, tickData, onToggleSim, onInjectNews }) => {
    const [stockName, setStockName] = useState('TCS');
    const [initPrice, setInitPrice] = useState('190.00');
    const [sector, setSector] = useState('TECH');
    const [newsText, setNewsText] = useState('');

    const handleToggle = () => {
        onToggleSim({ stock_name: stockName, start_price: initPrice, sector });
    };

    const handleInject = () => {
        if (!isRunning) {
            alert("Please start the simulation first!");
            return;
        }
        if (newsText.trim()) {
            onInjectNews({ headline: newsText, sector, current_price: lastKnownPrice });
            setNewsText('');
        }
    };

    const formatTime = (unixTime) => {
        if (!unixTime) return "09:15";
        const dateObj = new Date(unixTime * 1000);
        const hours = String(dateObj.getUTCHours()).padStart(2, '0');
        const minutes = String(dateObj.getUTCMinutes()).padStart(2, '0');
        return `${hours}:${minutes}`;
    };

    return (
        <div className="header-bar">
            <div style={{ fontWeight: 'bold', fontSize: '18px', display: 'flex', alignItems: 'center', gap: '10px' }}>
                <div style={{ width: '12px', height: '12px', background: 'var(--up-color)', borderRadius: '50%', boxShadow: '0 0 8px var(--up-color)' }}></div>
                CHRONOS ENGINE
            </div>
            <div className="controls">
                <input type="text" placeholder="Symbol" value={stockName} onChange={(e) => setStockName(e.target.value)} style={{ width: '70px' }} />
                <input type="number" placeholder="Init Price" value={initPrice} onChange={(e) => setInitPrice(e.target.value)} style={{ width: '80px' }} />
                <select value={sector} onChange={(e) => setSector(e.target.value)}>
                    <option value="TECH">TECH</option>
                    <option value="DEFENSE">DEFENSE</option>
                    <option value="PHARMA">PHARMA</option>
                    <option value="OIL">OIL / ENERGY</option>
                </select>
                <button onClick={handleToggle} style={{ backgroundColor: isRunning ? 'var(--text-muted)' : '#2962ff' }}>
                    {isRunning ? "PAUSE SIMULATION" : "START SIMULATION"}
                </button>
            </div>
            <div className="controls" style={{ borderLeft: '1px solid var(--border-color)', paddingLeft: '15px' }}>
                <input type="text" placeholder="Enter Breaking News..." value={newsText} onChange={(e) => setNewsText(e.target.value)} style={{ width: '300px' }} />
                <button className="inject" onClick={handleInject}>INJECT NEWS</button>
            </div>
            <div className="stat-box">
                <div>DAY: <span className="stat-val" style={{ color: '#ff9800' }}>{tickData?.day_count || 1}</span></div>
                <div>TIME: <span className="stat-val" style={{ color: '#ff9800' }}>{formatTime(tickData?.unix_time)}</span></div>
                <div>PRICE: <span className="stat-val">${(lastKnownPrice || 0).toFixed(2)}</span></div>
            </div>
        </div>
    );
};

export default Header;
