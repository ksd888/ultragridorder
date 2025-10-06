import asyncio
import json
from typing import Callable

class BinanceWebSocketClient:
    """WebSocket client for real-time data"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol.lower().replace('/', '')
        self.ws_url = f"wss://stream.binance.com:9443/ws/{self.symbol}@depth20@100ms"
        self.trade_url = f"wss://stream.binance.com:9443/ws/{self.symbol}@trade"
        
    async def connect_orderbook(self, callback: Callable):
        """Connect to orderbook stream"""
        import websockets
        
        async with websockets.connect(self.ws_url) as websocket:
            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    # Parse orderbook update
                    orderbook = {
                        'bids': [[float(p), float(q)] for p, q in data['bids']],
                        'asks': [[float(p), float(q)] for p, q in data['asks']],
                        'timestamp': data['E']
                    }
                    
                    # Call the callback
                    callback(orderbook)
                    
                except Exception as e:
                    print(f"WebSocket error: {e}")
                    await asyncio.sleep(5)
                    
    async def connect_trades(self, callback: Callable):
        """Connect to trades stream"""
        import websockets
        
        async with websockets.connect(self.trade_url) as websocket:
            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    # Parse trade
                    trade = {
                        'id': data['t'],
                        'price': float(data['p']),
                        'amount': float(data['q']),
                        'side': 'sell' if data['m'] else 'buy',
                        'timestamp': data['T']
                    }
                    
                    # Call the callback
                    callback(trade)
                    
                except Exception as e:
                    print(f"WebSocket error: {e}")
                    await asyncio.sleep(5)