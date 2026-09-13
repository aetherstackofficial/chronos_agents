import React, { useEffect, useRef } from 'react';
import { createChart, CrosshairMode, CandlestickSeries, HistogramSeries } from 'lightweight-charts';

const Chart = ({ data }) => {
    const chartContainerRef = useRef(null);
    const chartRef = useRef(null);
    const candleSeriesRef = useRef(null);
    const volumeSeriesRef = useRef(null);

    useEffect(() => {
        if (!chartContainerRef.current) return;

        const chartProperties = { 
            layout: { background: { type: 'solid', color: '#131722' }, textColor: '#d1d4dc' }, 
            grid: { vertLines: { color: '#2a2e39' }, horzLines: { color: '#2a2e39' } }, 
            crosshair: { mode: CrosshairMode.Normal }, 
            rightPriceScale: { borderColor: '#2a2e39' }, 
            timeScale: { borderColor: '#2a2e39', timeVisible: true, secondsVisible: false } 
        };
        
        const chart = createChart(chartContainerRef.current, chartProperties);
        chartRef.current = chart;

        const candleSeries = chart.addSeries(CandlestickSeries, { 
            upColor: '#26a69a', downColor: '#ef5350', borderVisible: false, 
            wickUpColor: '#26a69a', wickDownColor: '#ef5350' 
        });
        candleSeriesRef.current = candleSeries;
        
        const volumeSeries = chart.addSeries(HistogramSeries, {
            priceFormat: { type: 'volume' },
            priceScaleId: '', 
        });
        volumeSeriesRef.current = volumeSeries;

        chart.priceScale('').applyOptions({
            scaleMargins: { top: 0.8, bottom: 0 },
        });

        const handleResize = () => {
            if (chartContainerRef.current) {
                chart.applyOptions({ 
                    width: chartContainerRef.current.clientWidth, 
                    height: chartContainerRef.current.clientHeight 
                });
            }
        };

        window.addEventListener('resize', handleResize);
        setTimeout(handleResize, 100);

        return () => {
            window.removeEventListener('resize', handleResize);
            chart.remove();
        };
    }, []);

    useEffect(() => {
        if (data && candleSeriesRef.current && volumeSeriesRef.current) {
            try {
                candleSeriesRef.current.update({ 
                    time: data.unix_time, 
                    open: data.ohlc.open, 
                    high: data.ohlc.high, 
                    low: data.ohlc.low, 
                    close: data.ohlc.close 
                });
                volumeSeriesRef.current.update({ 
                    time: data.unix_time, 
                    value: data.step_volume, 
                    color: data.ohlc.close >= data.ohlc.open ? 'rgba(38, 166, 154, 0.4)' : 'rgba(239, 83, 80, 0.4)' 
                });
            } catch (err) {
                console.warn("Chart update skipped due to error:", err);
            }
        }
    }, [data]);

    return (
        <div className="panel">
            <div className="panel-title">Asset Price & Volume</div>
            <div ref={chartContainerRef} style={{ flex: 1, width: '100%', position: 'relative' }}></div>
        </div>
    );
};

export default Chart;
