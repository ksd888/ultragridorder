#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive Grid Allocation System
Dynamically adjusts grid allocation based on:
- Market volatility (ATR)
- Distance from current price
- Order flow strength (optional)

Features:
- Volatility-aware allocation
- Distance-based weighting (near/mid/far zones)
- Reversible weighting schemes (near_heavier / far_heavier)
- Respects exchange limits
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Literal, Optional
from dataclasses import dataclass

WeightScheme = Literal["near_heavier", "far_heavier", "equal"]


@dataclass
class GridLevel:
    """Single grid level configuration"""
    buy_price: float
    coin_size: float
    tp_price: float
    tp_pct: float
    notional: float
    zone: str  # "near", "mid", "far"
    weight: float


class AdaptiveGridAllocator:
    """
    Adaptive grid allocation with volatility and distance-based weighting

    Args:
        grid_levels: List of grid buy prices
        current_price: Current market price
        budget_usd: Total budget
        tp_pct: Take profit percentage
        weight_scheme: 'near_heavier', 'far_heavier', or 'equal'
        alpha: Weight distribution steepness (0.5-2.0)
        volatility: Market volatility (for adaptive sizing)
    """

    def __init__(self,
                 grid_levels: List[float],
                 current_price: float,
                 budget_usd: float,
                 tp_pct: float = 0.015,
                 weight_scheme: WeightScheme = "far_heavier",
                 alpha: float = 0.7,
                 volatility: Optional[float] = None,
                 min_notional: float = 10.0,
                 min_qty: float = 0.0001,
                 tick_size: float = 0.000001,
                 lot_size: float = 0.0001):

        self.grid_levels = sorted(grid_levels)
        self.current_price = current_price
        self.budget_usd = budget_usd
        self.tp_pct = tp_pct
        self.weight_scheme = weight_scheme
        self.alpha = alpha
        self.volatility = volatility or 0.03  # Default 3%

        self.min_notional = min_notional
        self.min_qty = min_qty
        self.tick_size = tick_size
        self.lot_size = lot_size

        # Calculate price range
        self.min_price = min(grid_levels)
        self.max_price = max(grid_levels)

        # Zone thresholds
        self.zone_near_pct = 0.05  # 5% from current price
        self.zone_mid_pct = 0.15   # 15% from current price

        print(f"Adaptive Grid Allocator Initialized")
        print(f"  Levels: {len(grid_levels)}")
        print(f"  Price range: ${self.min_price:.6f} - ${self.max_price:.6f}")
        print(f"  Current price: ${current_price:.6f}")
        print(f"  Budget: ${budget_usd:.2f}")
        print(f"  Weight scheme: {weight_scheme}")
        print(f"  Volatility: {self.volatility * 100:.2f}%")

    def classify_zone(self, level: float) -> str:
        """
        Classify grid level into zone based on distance from current price

        Returns:
            'near', 'mid', or 'far'
        """
        distance = abs(level - self.current_price) / self.current_price

        if distance <= self.zone_near_pct:
            return 'near'
        elif distance <= self.zone_mid_pct:
            return 'mid'
        else:
            return 'far'

    def calculate_distance_ratio(self, level: float) -> float:
        """
        Calculate normalized distance ratio [0, 1]

        0 = at current price
        1 = at price range boundary
        """
        if level <= self.current_price:
            # Below current price
            denom = max(self.current_price - self.min_price, 1e-12)
            distance = (self.current_price - level) / denom
        else:
            # Above current price
            denom = max(self.max_price - self.current_price, 1e-12)
            distance = (level - self.current_price) / denom

        return np.clip(distance, 0.0, 1.0)

    def calculate_weights(self) -> np.ndarray:
        """
        Calculate allocation weights for each grid level

        Returns:
            Array of weights (sum = budget_usd)
        """
        n = len(self.grid_levels)

        if self.weight_scheme == 'equal':
            # Equal allocation
            weights = np.ones(n) / n * self.budget_usd

        else:
            # Distance-based weighting
            distances = np.array([
                self.calculate_distance_ratio(level)
                for level in self.grid_levels
            ])

            if self.weight_scheme == 'near_heavier':
                # More weight to levels near current price
                # r = (1 - d)^alpha
                raw_weights = np.power(1.0 - distances, self.alpha)

            else:  # far_heavier
                # More weight to levels far from current price
                # r = d^alpha
                raw_weights = np.power(distances, self.alpha)

            # Normalize to budget
            if np.sum(raw_weights) > 1e-12:
                weights = raw_weights / np.sum(raw_weights) * self.budget_usd
            else:
                weights = np.ones(n) / n * self.budget_usd

        # Apply volatility adjustment (higher vol = more even distribution)
        if self.volatility > 0.05:  # High volatility (>5%)
            # Blend toward equal weighting
            equal_weights = np.ones(n) / n * self.budget_usd
            vol_factor = min((self.volatility - 0.05) / 0.05, 0.3)  # Max 30% blend
            weights = (1 - vol_factor) * weights + vol_factor * equal_weights

        return weights

    def allocate(self) -> List[GridLevel]:
        """
        Generate adaptive grid allocation

        Returns:
            List of GridLevel objects
        """
        print("\nCalculating adaptive allocation...")

        weights = self.calculate_weights()

        grid_output = []

        for level, allocated_usd in zip(self.grid_levels, weights):
            # Calculate coin size
            coin_size = allocated_usd / level

            # Round to lot size
            coin_size = self._round_to_lot(coin_size)

            # Calculate actual notional
            notional = coin_size * level

            # Check minimums
            if notional < self.min_notional or coin_size < self.min_qty:
                continue

            # Calculate TP
            tp_price = level * (1 + self.tp_pct)
            tp_price = self._round_to_tick(tp_price)

            # Classify zone
            zone = self.classify_zone(level)

            # Calculate weight (for reporting)
            weight = allocated_usd / self.budget_usd

            grid_output.append(GridLevel(
                buy_price=level,
                coin_size=coin_size,
                tp_price=tp_price,
                tp_pct=self.tp_pct,
                notional=notional,
                zone=zone,
                weight=weight
            ))

        # Adjust to match budget exactly
        total_allocated = sum(g.notional for g in grid_output)

        if total_allocated > 0 and abs(total_allocated - self.budget_usd) / self.budget_usd > 0.05:
            # More than 5% off - rescale
            scale_factor = self.budget_usd / total_allocated

            for grid_level in grid_output:
                grid_level.coin_size *= scale_factor
                grid_level.coin_size = self._round_to_lot(grid_level.coin_size)
                grid_level.notional = grid_level.buy_price * grid_level.coin_size

        print(f"[OK] Generated {len(grid_output)} valid allocations")

        return grid_output

    def _round_to_tick(self, price: float) -> float:
        """Round price to tick size (floor for buy, ceiling for sell)"""
        if self.tick_size <= 0:
            return price

        # For TP (sell), use ceiling
        return float(np.ceil(price / self.tick_size) * self.tick_size)

    def _round_to_lot(self, quantity: float) -> float:
        """Round quantity to lot size (floor)"""
        if self.lot_size <= 0:
            return quantity

        return float(np.floor(quantity / self.lot_size) * self.lot_size)

    def to_dataframe(self, grid_levels: List[GridLevel]) -> pd.DataFrame:
        """Convert grid levels to DataFrame"""
        return pd.DataFrame([
            {
                'buy_price': g.buy_price,
                'coin_size': g.coin_size,
                'tp_price': g.tp_price,
                'tp_pct': g.tp_pct,
                'notional': g.notional,
                'zone': g.zone,
                'weight': g.weight
            }
            for g in grid_levels
        ])

    def print_summary(self, grid_levels: List[GridLevel]):
        """Print allocation summary"""
        df = self.to_dataframe(grid_levels)

        print("\n" + "=" * 60)
        print("ADAPTIVE GRID ALLOCATION SUMMARY")
        print("=" * 60)

        # Overall stats
        print(f"Total levels: {len(grid_levels)}")
        print(f"Total allocated: ${df['notional'].sum():.2f}")
        print(f"Avg per level: ${df['notional'].mean():.2f}")
        print(f"Min allocation: ${df['notional'].min():.2f}")
        print(f"Max allocation: ${df['notional'].max():.2f}")

        # Zone breakdown
        print("\nZone breakdown:")
        for zone in ['near', 'mid', 'far']:
            zone_df = df[df['zone'] == zone]
            if not zone_df.empty:
                print(f"  {zone.upper()}: {len(zone_df)} levels, ${zone_df['notional'].sum():.2f} ({zone_df['notional'].sum() / df['notional'].sum() * 100:.1f}%)")

        print("=" * 60)


# ================= Example Usage =================

if __name__ == '__main__':
    # Example: Create adaptive allocation for existing grid

    # Sample grid levels (from CSV or manual)
    grid_levels = [
        0.4500, 0.4600, 0.4700, 0.4800, 0.4900,
        0.5000, 0.5100, 0.5200, 0.5300, 0.5400,
        0.5500, 0.5600, 0.5700, 0.5800, 0.5900
    ]

    current_price = 0.5200
    budget = 300.0

    # Create allocator with "far_heavier" strategy
    allocator = AdaptiveGridAllocator(
        grid_levels=grid_levels,
        current_price=current_price,
        budget_usd=budget,
        tp_pct=0.015,
        weight_scheme='far_heavier',  # More allocation to far levels
        alpha=0.7,
        volatility=0.03,  # 3% volatility
        min_notional=10.0,
        tick_size=0.0001,
        lot_size=1.0
    )

    # Generate allocation
    allocated_grid = allocator.allocate()

    # Print summary
    allocator.print_summary(allocated_grid)

    # Save to CSV
    df = allocator.to_dataframe(allocated_grid)
    df.to_csv('grid_adaptive.csv', index=False)

    print("\n[SUCCESS] Adaptive allocation saved to grid_adaptive.csv")
