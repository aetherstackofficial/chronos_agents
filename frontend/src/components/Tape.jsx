import React from 'react';

const Tape = ({ trades }) => {
    return (
        <div className="panel">
            <div className="panel-title">Tape (Recent Trades)</div>
            <div className="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Price</th>
                            <th>Size</th>
                            <th>Buyer</th>
                            <th>Seller</th>
                        </tr>
                    </thead>
                    <tbody>
                        {(trades || []).map((t, i) => (
                            <tr key={i}>
                                <td style={{ color: 'var(--text-muted)' }}>{t.time}</td>
                                <td className={Math.random() > 0.5 ? 'positive' : 'negative'}>{t.price.toFixed(2)}</td>
                                <td>{t.size}</td>
                                <td>{t.buyer}</td>
                                <td>{t.seller}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default Tape;
