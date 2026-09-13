import React from 'react';

const formatMoney = (num) => '$' + parseFloat(num).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });

const Leaderboard = ({ agents }) => {
    return (
        <div className="panel">
            <div className="panel-title">Live Agent PnL Leaderboard</div>
            <div className="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Agent ID</th>
                            <th>Type</th>
                            <th>Net PnL</th>
                            <th>Inventory</th>
                        </tr>
                    </thead>
                    <tbody>
                        {(agents || []).map((a, i) => (
                            <tr key={i}>
                                <td>{a.id}</td>
                                <td style={{ color: 'var(--text-muted)' }}>{a.type}</td>
                                <td className={a.pnl >= 0 ? 'positive' : 'negative'}>
                                    {a.pnl > 0 ? '+' : ''}{formatMoney(a.pnl)}
                                </td>
                                <td className={a.pos >= 0 ? 'positive' : 'negative'}>
                                    {a.pos}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default Leaderboard;
