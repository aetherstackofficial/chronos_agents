import React from 'react';

const formatMoney = (num) => '$' + parseFloat(num).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });

const OrderBook = ({ book }) => {
    const maxBookSize = 5000;
    
    const asks = (book?.asks || []).slice().reverse();
    const bids = book?.bids || [];
    const spread = book?.spread || 0;

    return (
        <div className="panel">
            <div className="panel-title">Limit Order Book</div>
            <div className="order-book-container">
                <div className="ob-header"><span>Price</span><span>Size</span></div>
                <div className="ob-rows" style={{ display: 'flex', flexDirection: 'column-reverse' }}>
                    {asks.map((a, i) => (
                        <div key={'ask-'+i} className="ob-row ask-row">
                            <div className="depth-bar" style={{ width: `${Math.min(100, (a.size / maxBookSize) * 100)}%` }}></div>
                            <span className="ask-price">{a.price.toFixed(2)}</span>
                            <span>{a.size}</span>
                        </div>
                    ))}
                </div>
                <div className="spread-row">SPREAD: {formatMoney(spread)}</div>
                <div className="ob-rows">
                    {bids.map((b, i) => (
                        <div key={'bid-'+i} className="ob-row bid-row">
                            <div className="depth-bar" style={{ width: `${Math.min(100, (b.size / maxBookSize) * 100)}%` }}></div>
                            <span className="bid-price">{b.price.toFixed(2)}</span>
                            <span>{b.size}</span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default OrderBook;
