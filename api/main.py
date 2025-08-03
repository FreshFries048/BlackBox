"""
Production-grade API for BlackBox Trading System
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import pandas as pd
import uvicorn

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import directly from the module file
import blackbox_core as bc
from blackbox_core_pkg.result_writer import ResultWriter
from blackbox_core_pkg.risk import TradingCosts, EnhancedRiskManager
from blackbox_core_pkg.exceptions import MissingFeatureError
from node_detector import NodeDetectorEngine
from trade_executor import TradeExecutorEngine

NodeEngine = bc.NodeEngine

app = FastAPI(title="BlackBox Trading API", version="1.0.0")


class BacktestRequest(BaseModel):
    """Request model for backtest endpoint."""
    data_file: str = Field(..., description="Path to market data file")
    rr_multiple: float = Field(2.0, ge=0.1, le=10.0, description="Risk-reward multiple")
    commission: float = Field(0.0, ge=0.0, le=1.0, description="Commission percentage")
    spread: float = Field(0.0, ge=0.0, description="Spread in points")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")


class BacktestSummary(BaseModel):
    """Summary statistics for a backtest run."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: Optional[float] = None


class BacktestResponse(BaseModel):
    """Response model for backtest endpoint."""
    run_id: str
    summary: BacktestSummary
    trades_url: str
    signals_url: str
    metadata: Dict[str, Any]


class InsightsResponse(BaseModel):
    """Response model for insights endpoint."""
    run_id: str
    seasonal_stats: Dict[str, Any]
    node_performance: Dict[str, Any]
    risk_metrics: Dict[str, Any]


# Global instances
result_writer = ResultWriter()
background_tasks_status = {}


def calculate_summary_stats(trades_df: pd.DataFrame) -> BacktestSummary:
    """Calculate summary statistics from trades DataFrame."""
    if trades_df.empty:
        return BacktestSummary(
            total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0.0, total_pnl=0.0, avg_win=0.0, avg_loss=0.0,
            profit_factor=0.0, max_drawdown=0.0
        )
    
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl_points'] > 0])
    losing_trades = len(trades_df[trades_df['pnl_points'] < 0])
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    
    total_pnl = trades_df['pnl_points'].sum()
    avg_win = trades_df[trades_df['pnl_points'] > 0]['pnl_points'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df['pnl_points'] < 0]['pnl_points'].mean() if losing_trades > 0 else 0
    
    gross_profit = trades_df[trades_df['pnl_points'] > 0]['pnl_points'].sum()
    gross_loss = abs(trades_df[trades_df['pnl_points'] < 0]['pnl_points'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Calculate max drawdown
    cumulative_pnl = trades_df['pnl_points'].cumsum()
    running_max = cumulative_pnl.expanding().max()
    drawdown = (cumulative_pnl - running_max)
    max_drawdown = abs(drawdown.min()) if not drawdown.empty else 0
    
    return BacktestSummary(
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        total_pnl=total_pnl,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        max_drawdown=max_drawdown
    )


def run_backtest_job(request: BacktestRequest, run_id: str):
    """Background job to run backtest."""
    try:
        background_tasks_status[run_id] = {"status": "running", "progress": 0}
        
        # Load data
        data_path = Path(request.data_file)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Load market data
        df = pd.read_csv(data_path)
        if request.start_date:
            df = df[df['time'] >= request.start_date]
        if request.end_date:
            df = df[df['time'] <= request.end_date]
        
        # Prepare data dictionary
        market_data = {
            'timestamp': pd.to_datetime(df['time']),
            'price': df['close'],
            'open': df['open'],
            'high': df['high'],
            'low': df['low'],
            'volume': df.get('tick_volume', df.get('volume', 1000)),
            'spread': df.get('spread', 0.0001)
        }
        
        background_tasks_status[run_id]["progress"] = 25
        
        # Load strategy nodes
        nodes_dir = Path("blackbox_nodes")
        node_engine = NodeEngine()
        node_engine.load_nodes_from_folder(str(nodes_dir))
        
        # Initialize components with trading costs
        trading_costs = TradingCosts(
            commission_perc=request.commission,
            spread_points=request.spread
        )
        
        risk_manager = EnhancedRiskManager(
            rr_multiple=request.rr_multiple,
            trading_costs=trading_costs
        )
        
        background_tasks_status[run_id]["progress"] = 50
        
        # Run signal detection
        detector = NodeDetectorEngine(node_engine, market_data)
        signals = detector.run_detection(live_output=False)
        
        # Filter tradeable signals
        tradeable_signals = [s for s in signals if any(conf in s.confidence for conf in ["High", "Highest"])]
        
        background_tasks_status[run_id]["progress"] = 75
        
        # Run backtest
        executor = TradeExecutorEngine(market_data, risk_manager)
        executor.run_backtest(tradeable_signals)
        
        # Prepare DataFrames
        trades_data = []
        for trade in executor.closed_trades:
            trades_data.append({
                'node_name': trade.node_name,
                'timestamp': trade.timestamp,
                'side': trade.side,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'pnl_points': trade.pnl_points,
                'pnl_percent': trade.pnl_percent,
                'exit_reason': trade.exit_reason,
                'duration_held': trade.duration_held,
                'confidence': trade.confidence
            })
        
        trades_df = pd.DataFrame(trades_data)
        
        signals_data = []
        for signal in signals:
            signals_data.append({
                'node_name': signal.node_name,
                'timestamp': signal.timestamp,
                'confidence': signal.confidence,
                'signal_strength': signal.signal_strength,
                'market_condition': signal.market_condition
            })
        
        signals_df = pd.DataFrame(signals_data)
        
        # Save results
        metadata = {
            'rr_multiple': request.rr_multiple,
            'commission_perc': request.commission,
            'spread_points': request.spread,
            'start_date': request.start_date,
            'end_date': request.end_date,
            'total_signals': len(signals),
            'tradeable_signals': len(tradeable_signals),
            'data_file': str(data_path)
        }
        
        result_writer.save_run(
            trades_df=trades_df,
            signals_df=signals_df,
            metadata=metadata,
            data_file_path=data_path,
            node_dir_path=nodes_dir
        )
        
        background_tasks_status[run_id] = {
            "status": "completed",
            "progress": 100,
            "summary": calculate_summary_stats(trades_df)
        }
        
    except Exception as e:
        background_tasks_status[run_id] = {
            "status": "failed",
            "error": str(e),
            "progress": 0
        }


@app.post("/backtest", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """Run a backtest with specified parameters."""
    import uuid
    run_id = str(uuid.uuid4())
    
    # Start background task
    background_tasks.add_task(run_backtest_job, request, run_id)
    
    # Return immediate response
    return BacktestResponse(
        run_id=run_id,
        summary=BacktestSummary(
            total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0.0, total_pnl=0.0, avg_win=0.0, avg_loss=0.0,
            profit_factor=0.0, max_drawdown=0.0
        ),
        trades_url=f"/results/{run_id}/trades.parquet",
        signals_url=f"/results/{run_id}/signals.parquet",
        metadata={"status": "processing"}
    )


@app.get("/backtest/{run_id}/status")
async def get_backtest_status(run_id: str):
    """Get the status of a running backtest."""
    if run_id not in background_tasks_status:
        raise HTTPException(status_code=404, detail="Run ID not found")
    
    return background_tasks_status[run_id]


@app.get("/insights/{run_id}", response_model=InsightsResponse)
async def get_insights(run_id: str):
    """Get insights and analysis for a completed backtest run."""
    try:
        run_data = result_writer.load_run(run_id)
        trades_df = run_data['trades_df']
        
        # Seasonal analysis
        trades_df['hour'] = pd.to_datetime(trades_df['timestamp']).dt.hour
        trades_df['day_of_week'] = pd.to_datetime(trades_df['timestamp']).dt.dayofweek
        
        seasonal_stats = {
            'hourly_win_rate': trades_df.groupby('hour')['pnl_points'].apply(
                lambda x: (x > 0).mean() * 100
            ).to_dict(),
            'daily_win_rate': trades_df.groupby('day_of_week')['pnl_points'].apply(
                lambda x: (x > 0).mean() * 100
            ).to_dict()
        }
        
        # Node performance
        node_performance = {}
        for node in trades_df['node_name'].unique():
            node_trades = trades_df[trades_df['node_name'] == node]
            node_performance[node] = {
                'total_trades': len(node_trades),
                'win_rate': (node_trades['pnl_points'] > 0).mean() * 100,
                'avg_pnl': node_trades['pnl_points'].mean(),
                'total_pnl': node_trades['pnl_points'].sum()
            }
        
        # Risk metrics
        risk_metrics = {
            'max_consecutive_losses': 0,  # Would need more complex calculation
            'profit_factor': calculate_summary_stats(trades_df).profit_factor,
            'sharpe_ratio': None  # Would need returns series
        }
        
        return InsightsResponse(
            run_id=run_id,
            seasonal_stats=seasonal_stats,
            node_performance=node_performance,
            risk_metrics=risk_metrics
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/results/{run_id}/trades.parquet")
async def get_trades_parquet(run_id: str):
    """Download trades data as Parquet file."""
    trades_path = Path(f"results/{run_id}/trades.parquet")
    if not trades_path.exists():
        raise HTTPException(status_code=404, detail="Trades file not found")
    
    from fastapi.responses import FileResponse
    return FileResponse(trades_path, media_type="application/octet-stream")


@app.get("/results/{run_id}/signals.parquet")
async def get_signals_parquet(run_id: str):
    """Download signals data as Parquet file."""
    signals_path = Path(f"results/{run_id}/signals.parquet")
    if not signals_path.exists():
        raise HTTPException(status_code=404, detail="Signals file not found")
    
    from fastapi.responses import FileResponse
    return FileResponse(signals_path, media_type="application/octet-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
