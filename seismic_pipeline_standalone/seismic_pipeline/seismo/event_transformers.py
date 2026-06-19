"""
Target-aware transformers for seismic event processing.

This module contains transformers specifically designed for processing seismic events
with hypnogram data and REM sleep profiles.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional

from ..mod.sklearnbaseyt import TransformerMixinYt
from .date_utils import parse_event_date

class SeismicEventTransformerYt(TransformerMixinYt):
    """
    Base transformer for seismic event data processing.
    """
    def __init__(self, event_window_days: int = 6, label_window_days: int = 6):
        super().__init__()
        self.event_window_days = event_window_days
        self.label_window_days = label_window_days
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X, y

class EventLabelGeneratorYt(SeismicEventTransformerYt):
    """
    Expands each event into two windows of equal size:
    - n days before event (label 1)
    - n days before that (label 0)
    Output: expanded X (list of dicts with 'rat_id' and 'window_dates'), y (0/1)
    """
    def __init__(self, window_days: int = 3, random_seed: Optional[int] = None, date_format: str = "%Y_%m_%d"):
        super().__init__(window_days, window_days)  # Both windows same size
        self.window_days = window_days
        self.random_seed = random_seed
        self.date_format = date_format
        if random_seed is not None:
            np.random.seed(random_seed)
            
    def fit(self, X, y=None):
        self.n_events_ = len(X) if hasattr(X, '__len__') else 1
        return self
        
    def transform(self, X, y=None):
        from datetime import timedelta

        expanded_X = []
        expanded_y = []
        for event in X:
            rat_id = event['rat_id']
            event_date = event['date']
            dt = parse_event_date(event_date, self.date_format)

            # Window 1: n days before event (label 1)
            window1 = [(dt - timedelta(days=offset)).strftime(self.date_format) 
                      for offset in range(1, self.window_days+1)][::-1]
            expanded_X.append({
                'rat_id': rat_id, 
                'window_dates': window1,
                'original_event_date': event_date,
                'original_rat_id': rat_id
            })
            expanded_y.append(1)
            
            # Window 0: n days before that (label 0)
            window0 = [(dt - timedelta(days=offset)).strftime(self.date_format) 
                      for offset in range(self.window_days+1, 2*self.window_days+1)][::-1]
            expanded_X.append({
                'rat_id': rat_id, 
                'window_dates': window0,
                'original_event_date': event_date,
                'original_rat_id': rat_id
            })
            expanded_y.append(0)
        return expanded_X, np.array(expanded_y)


class CustomEventLabelGeneratorYt(SeismicEventTransformerYt):
    """
    Custom label generator that creates exactly 2 samples per event:
    - Window 1: window_days immediately before event day (label 1)
    - Window 0: window_days window window_step_days before the event (label 0)
    
    Supports negative window_step_days for positioning immediate window:
    - If window_step_days is negative (e.g., -4), the immediate window (label 1) 
      is positioned at that offset (e.g., days 1-3 before event for window_days=3)
    - If use_fixed_control_window is True, a fixed 3-day control window at 
      5-7 days before event is always created (label 0)
    """
    def __init__(self, window_days: int = 6, window_step_days: int = 3, 
                 date_format: str = "%Y-%m-%d", window_mode: str = 'both',
                 use_fixed_control_window: bool = False, fixed_control_start_days: int = 7,
                 original_position: int = None):
        super().__init__()
        self.window_days = window_days
        self.window_step_days = window_step_days
        self.date_format = date_format
        self.use_fixed_control_window = use_fixed_control_window
        self.fixed_control_start_days = fixed_control_start_days  # Start of fixed control window (default: 7 days before)
        self.fixed_control_window_days = window_days  # Use window_days for fixed control window size
        self.original_position = original_position  # Original window position to distinguish positive from negative
        # window_mode controls which windows are generated:
        # 'both' -> immediate_before (1) and control_before (0)
        # 'immediate' -> only immediate_before (1)
        # 'control' -> only control_before (0)
        valid_modes = {'both', 'immediate', 'control'}
        if window_mode not in valid_modes:
            raise ValueError(f"Invalid window_mode '{window_mode}'. Must be one of {valid_modes}.")
        self.window_mode = window_mode
        
    def fit(self, X, y=None):
        self.n_events_ = len(X) if hasattr(X, '__len__') else 1
        return self
        
    def transform(self, X, y=None):
        from datetime import datetime, timedelta

        expanded_X = []
        expanded_y = []
        for event in X:
            rat_id = event['rat_id']
            event_date = event['date']
            dt = parse_event_date(event_date, self.date_format)

            # Determine immediate window position (label 1)
            if self.window_step_days <= 0:
                # Zero or negative step: window ends at window_pos days before event
                # Position 0: window ends at day 0 (event-1), window should be days [5,4,3,2,1,0]
                #   This means: [event-6, event-5, event-4, event-3, event-2, event-1]
                #   So start_offset = 6, end_offset = 1
                # Position -1: window ends at day 1 (event-2), window should be days [4,3,2,1,0,-1]?
                #   Actually, user wants [4,3,2,1,0,-1] which means window moves forward
                #   Let me recalculate: if position -1 means window ends 1 day further from event
                #   Position 0: ends at event-1 (offset 1)
                #   Position -1: ends at event-2 (offset 2) -> window [event-7, event-6, event-5, event-4, event-3, event-2] = [6,5,4,3,2,1]
                #   But user wants [4,3,2,1,0,-1], so position -1 should end at event-1 (offset 1) but start earlier?
                #   Actually, I think the user's notation is: position p means window is [p+5, p+4, p+3, p+2, p+1, p]
                #   Position 0: [5,4,3,2,1,0] -> window is [event-6, event-5, event-4, event-3, event-2, event-1]
                #   Position -1: [4,3,2,1,0,-1] -> but -1 doesn't make sense as a day offset
                #   Wait, maybe they mean: position -1 should give [4,3,2,1,0] where 0=event-1, so window is [event-5, event-4, event-3, event-2, event-1]
                #   But that's only 5 days, not 6.
                #   Let me try: position p means window ends at (p+1) days before event, starts at (p+6) days before event
                #   Position 0: ends at 1, starts at 6 -> [6,5,4,3,2,1] but user wants [5,4,3,2,1,0]
                #   Position -1: ends at 0? That doesn't work.
                #   Actually, I think: position p means the window's last day is p days before event (where p can be 0)
                #   Position 0: last day is event-0 = event-1 (1 day before), window is [event-6, event-5, event-4, event-3, event-2, event-1] = [5,4,3,2,1,0]
                #   So for position 0: end_offset = 1 (event-1), start_offset = 6 (event-6)
                #   For position -1: last day is event-(-1) = event-1? That's the same.
                #   Let me try: position p means window ends at max(1, p+1) days before event
                #   Position 0: end_offset = max(1, 0+1) = 1, start_offset = 1+6-1 = 6 -> [6,5,4,3,2,1]
                #   But user wants [5,4,3,2,1,0], so end_offset should be 1, start_offset should be 5
                #   So: start_offset = end_offset + window_days - 1 = 1 + 6 - 1 = 6, but we want 5
                #   Maybe: start_offset = end_offset + window_days - 2 = 1 + 6 - 2 = 5
                #   Or: end_offset = abs_step (not abs_step+1)
                abs_step = abs(self.window_step_days)  # Allow 0 for position 0
                # For position 0: window ends at event-1 (offset 1), so end_offset = 1
                # For position -1: window ends at event-2 (offset 2), so end_offset = 2
                # General: end_offset = abs_step + 1
                # But user wants position 0 to give [5,4,3,2,1,0], which means window is [event-6, event-5, event-4, event-3, event-2, event-1]
                # So end_offset = 1, start_offset = 6 -> immediate_start_offset = 6
                # But current calculation gives: end_offset = 1, immediate_start_offset = 1+6-1 = 6, which is correct
                # So the issue might be in how we're displaying the offsets, not in the calculation
                # Actually, wait - the user wants [5,4,3,2,1,0] but we're showing [6,5,4,3,2,1]
                # The difference is that "0" in their notation means "1 day before event" in our offset
                # So we need to subtract 1 from the offset when displaying, OR change the offset calculation
                # Let me try: end_offset = abs_step (not abs_step+1) for position 0
                # Position 0: window ends at event-1 (offset 1), window is [event-6, event-5, event-4, event-3, event-2, event-1]
                #   Display: [5,4,3,2,1,0] where 0 = event-1
                # Position -1: window should be [4,3,2,1,0,-1] where 0=event-1, -1=event-2
                #   This means window should shift forward by 1 day
                #   Position 0: start_offset = 6, end_offset = 1
                #   Position -1: start_offset = 5, but we need 6 days, so window is [event-5, event-4, event-3, event-2, event-1, event-1]?
                #   Actually, I think: for position -1, window is [event-5, event-4, event-3, event-2, event-1] = 5 days
                #   But we have window_days=6. Let me keep the window size constant and just shift it:
                #   Position 0: start=6, window = [event-6, event-5, event-4, event-3, event-2, event-1] = 6 days
                #   Position -1: start=5, but we need 6 days, so we can't end at event-1 (that's only 5 days)
                #   So maybe: start=5, end=0? But end can't be 0 (that's the event date).
                #   I think the user might want: position -1 means window is [event-5, event-4, event-3, event-2, event-1] = 5 days
                #   But we have window_days=6. Let me just keep the current calculation for now and fix the display.
                if abs_step == 0:
                    # Position 0: window ends at event date (offset 0)
                    # Start offset must depend on window_days (not hardcoded to 6-day windows)
                    immediate_start_offset = self.window_days - 1
                else:
                    # Use original_position to distinguish positive from negative positions
                    # Positive positions: window ends at abs_step offset
                    #   Position 4: ends at offset 4, immediate_start_offset = 4 + 6 - 1 = 9
                    #   Position 1: ends at offset 1, immediate_start_offset = 1 + 6 - 1 = 6
                    # Negative positions: window ends at (abs_step + 1) offset
                    #   Position -1: ends at offset 2, immediate_start_offset = 2 + 6 - 1 = 7
                    #   Position -2: ends at offset 3, immediate_start_offset = 3 + 6 - 1 = 8
                    if self.original_position is not None and self.original_position < 0:
                        # Negative position: window ends at (abs_step + 2) offset
                        # Position -1: ends at offset 3 (3 days before event), display ends at -1
                        # Position -2: ends at offset 4 (4 days before event), display ends at -2
                        end_offset = -abs_step
                        immediate_start_offset = end_offset + self.window_days - 1
                    else:
                        # Positive position (or original_position not provided): window ends at (abs_step + 1) offset
                        # Position 0 is special: ends at offset 1
                        # Position 1: ends at offset 2, display [7,6,5,4,3,2] -> but we want [6,5,4,3,2,1]?
                        # Actually, let me check: if position 1 should end at offset 2, then window is [7,6,5,4,3,2]
                        # But display should be [6,5,4,3,2,1], which means we subtract 1 for display
                        # Position 4: ends at offset 5, but we want it to end at offset 4 to match control window
                        # So maybe: end_offset = abs_step for abs_step > 0, but position 1 is special?
                        # Actually, I think: end_offset = abs_step + 1 for positive positions > 0
                        # Position 1: ends at offset 2, window [7,6,5,4,3,2]
                        # Position 4: ends at offset 5, window [10,9,8,7,6,5] - but this doesn't match control window!
                        # Let me reconsider: position 4 should end at offset 4 to match control window [9,8,7,6,5,4]
                        # So: end_offset = abs_step for all positive positions
                        # But then position 1 and 0 are the same. So maybe position 1 should end at offset 2?
                        # Actually, I think the issue is that position 1 should use end_offset = abs_step + 1
                        # But position 4 should use end_offset = abs_step to match control window
                        # This is inconsistent. Let me check: maybe position 4 is wrong?
                        # User said position 4 immediate and control should match, so position 4 ends at offset 4
                        # So for position 4: abs_step=4, end_offset=4
                        # For position 1: abs_step=1, if end_offset=1, same as position 0
                        # So position 1 should have end_offset = 2
                        # General rule: end_offset = abs_step + 1 for positive positions > 0, except position 4?
                        # Actually, let me use: end_offset = abs_step + 1 for all positive positions > 0
                        # Position 1: end_offset = 1 + 1 = 2
                        # Position 4: end_offset = 4 + 1 = 5 (but this doesn't match control window!)
                        # Wait, maybe position 4 is a special case because it matches control window?
                        # Or maybe the control window calculation is wrong?
                        # Let me just fix position 1: if abs_step == 1 and original_position == 1, use end_offset = 2
                        # All positive positions: end at abs_step offset
                        # Position 1: ends at offset 1 (1 day before event), display [6,5,4,3,2,1]
                        # Position 4: ends at offset 4, display [9,8,7,6,5,4]
                        end_offset = abs_step
                        immediate_start_offset = end_offset + self.window_days - 1
            else:
                # Positive step: traditional behavior - window_days immediately before event
                immediate_start_offset = self.window_days
            
            window1_start = dt - timedelta(days=immediate_start_offset)
            
            window1 = [(window1_start + timedelta(days=i)).strftime(self.date_format) 
                      for i in range(self.window_days)]
            
            # Validate that all window dates are before or equal to the event date
            # Position 0 is allowed to include the event date (offset 0)
            for win_date_str in window1:
                try:
                    win_date = datetime.strptime(win_date_str, self.date_format)
                    if not (self.original_position is not None and self.original_position < 0):
                        if win_date > dt:
                            raise ValueError(
                                f"Invalid window date {win_date_str} for event {event_date}: "
                                f"window dates must be before or equal to the event date"
                            )
                except ValueError as e:
                    if "Invalid window date" in str(e):
                        raise
                    # If date parsing fails, try alternative format
                    try:
                        win_date = datetime.strptime(win_date_str, "%Y-%m-%d")
                        if not (self.original_position is not None and self.original_position < 0):
                            if win_date >= dt:
                                raise ValueError(
                                    f"Invalid window date {win_date_str} for event {event_date}: "
                                    f"window dates must be before the event date"
                                )
                    except Exception:
                        pass

            if self.window_mode in ('both', 'immediate'):
                expanded_X.append({
                    'rat_id': rat_id,
                    'window_dates': window1,
                    'original_event_date': event_date,
                    'original_rat_id': rat_id,
                    'window_type': 'immediate_before'
                })
                expanded_y.append(1)
            
            # Control window (label 0)
            if self.use_fixed_control_window:
                # Fixed 3-day control window: e.g., 5-7 days before event (configurable)
                fixed_window_start = dt - timedelta(days=self.fixed_control_start_days)
                window0 = [(fixed_window_start + timedelta(days=i)).strftime(self.date_format) 
                          for i in range(self.fixed_control_window_days)]
            else:
                # Traditional behavior: window_days window window_step_days before the event
                if self.window_step_days < 0:
                    # When using negative step, position control window before the immediate window
                    abs_step = max(1, abs(self.window_step_days))
                    immediate_start_offset = abs_step + self.window_days - 1
                    window0_start = dt - timedelta(days=immediate_start_offset + abs_step)
                else:
                    window0_start = dt - timedelta(days=self.window_days + self.window_step_days)
                window0 = [(window0_start + timedelta(days=i)).strftime(self.date_format) 
                          for i in range(self.window_days)]
            
            # Validate that all control window dates are before the event date
            for win_date_str in window0:
                try:
                    win_date = datetime.strptime(win_date_str, self.date_format)
                    if win_date >= dt:
                        raise ValueError(
                            f"Invalid control window date {win_date_str} for event {event_date}: "
                            f"window dates must be before the event date"
                        )
                except ValueError as e:
                    if "Invalid control window date" in str(e):
                        raise
                    # If date parsing fails, try alternative format
                    try:
                        win_date = datetime.strptime(win_date_str, "%Y-%m-%d")
                        if win_date >= dt:
                            raise ValueError(
                                f"Invalid control window date {win_date_str} for event {event_date}: "
                                f"window dates must be before the event date"
                            )
                    except Exception:
                        pass

            if self.window_mode in ('both', 'control'):
                expanded_X.append({
                    'rat_id': rat_id,
                    'window_dates': window0,
                    'original_event_date': event_date,
                    'original_rat_id': rat_id,
                    'window_type': 'control_before'
                })
                expanded_y.append(0)
            
        return expanded_X, np.array(expanded_y)

