"""
Specialized cache manager for hypnogram files from multiple sources.

This module provides caching functionality for hypnogram files from both
local/network drives and S3 storage, with focus on REM profile calculation.
"""

import os
from datetime import timedelta

from .date_utils import normalize_date_to_yyyymmdd, parse_event_date
import pickle
import hashlib
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union, Any, Callable
from pathlib import Path
import logging
import boto3
from datetime import datetime
import shutil
import tempfile


class HypnogramCacheManagerYt:
    """
    Specialized cache manager for hypnogram files from multiple sources.
    
    This class handles caching of hypnogram files from both local/network drives
    and S3 storage, with focus on REM profile calculation from cached data.
    """
    
    def __init__(self, 
                 local_cache_dir: str = './hypnogram_cache',
                 local_data_root: str = '/home/ponomattik/mnt/wd/rat',
                 s3_config: Optional[Dict] = None,
                 s3_rat_bucket: str = 'rat',
                 s3_temp_bucket: str = 'temp',
                 target_channels: List[str] = None,
                 sampling_rate: int = 250):
        """
        Initialize the hypnogram cache manager.
        
        Parameters
        ----------
        local_cache_dir : str, default='./hypnogram_cache'
            Local directory for caching hypnogram files
        local_data_root : str, default='/home/ponomattik/mnt/wd/rat'
            Root directory for local/network drive data
        s3_config : dict, optional
            S3 connection configuration
        s3_rat_bucket : str, default='rat'
            S3 bucket name for rat data
        s3_temp_bucket : str, default='temp'
            S3 bucket name for hypnogram files
        target_channels : list of str, optional
            Target channels to use for analysis
        sampling_rate : int, default=250
            Sampling rate of the data
        """
        self.local_cache_dir = Path(local_cache_dir)
        self.local_data_root = Path(local_data_root)
        self.s3_config = s3_config
        self.s3_rat_bucket = s3_rat_bucket
        self.s3_temp_bucket = s3_temp_bucket
        self.target_channels = target_channels or ['cxf', 'cxb', 'htl', 'hcm']
        self.sampling_rate = sampling_rate
        
        # Create cache directory
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # --------- S3 client handling -------------------------------------------------
        # We create the boto3 client lazily and keep only one instance per
        # unique configuration. This avoids repeatedly instantiating heavy
        # boto3 objects during GridSearch cloning and, more importantly,
        # makes the class picklable because the boto3 client itself cannot
        # be pickled.
        self._s3_client = None  # actual boto3 client (lazy)
        
        # Class-level cache: {frozenset(config_items): boto3_client}
        if not hasattr(HypnogramCacheManagerYt, "_CLIENT_CACHE"):
            HypnogramCacheManagerYt._CLIENT_CACHE = {}
        # -----------------------------------------------------------------------------
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Track which warnings we've already shown to avoid spam
        self._shown_warnings = set()
        
        # Negative cache: set of (rat_id, date) tuples known to be missing from
        # both local and S3. Prevents repeated S3 timeout attempts during grid
        # search fits.
        self._missing_negative_cache: set = set()
        
        # S3 reachability flag – once we detect the endpoint is unreachable we
        # stop making further requests for the lifetime of this instance.
        self._s3_reachable: Optional[bool] = None  # None = not tested yet
        
        # Cache index for tracking cached files
        self.cache_index_file = self.local_cache_dir / 'cache_index.pkl'
        self.cache_index = self._load_cache_index()
        
        # Migrate cache index to use normalized date formats
        self._migrate_cache_index()
        
    def _load_cache_index(self) -> Dict[str, Dict[str, Any]]:
        """Load cache index from disk."""
        if self.cache_index_file.exists():
            try:
                with open(self.cache_index_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache index: {e}")
        return {}
        
    def _save_cache_index(self):
        """Save cache index to disk."""
        try:
            with open(self.cache_index_file, 'wb') as f:
                pickle.dump(self.cache_index, f)
        except Exception as e:
            self.logger.error(f"Failed to save cache index: {e}")
    
    def _migrate_cache_index(self):
        """
        Migrate cache index to use normalized date formats (YYYY_MM_DD).
        This handles the transition from dash-separated dates to underscore-separated dates.
        """
        if not self.cache_index:
            return
        
        migrated = False
        new_index = {}
        
        for old_key, cache_info in self.cache_index.items():
            # Generate normalized key
            rat_id = cache_info.get('rat_id', '')
            date = cache_info.get('date', '')
            
            if not rat_id or not date:
                # Try to extract from old_key if info is missing
                if '_' in old_key:
                    parts = old_key.split('_', 1)
                    if len(parts) == 2:
                        rat_id = parts[0]
                        date = parts[1]
            
            if rat_id and date:
                normalized_key = self._generate_cache_key(rat_id, date)
                
                # Check if we already have this normalized key
                if normalized_key in new_index:
                    # Duplicate - keep the one with the most recent cache file
                    existing_info = new_index[normalized_key]
                    existing_file = Path(existing_info.get('cache_file', ''))
                    current_file = Path(cache_info.get('cache_file', ''))
                    
                    if current_file.exists() and existing_file.exists():
                        # Keep the newer file
                        if current_file.stat().st_mtime > existing_file.stat().st_mtime:
                            new_index[normalized_key] = cache_info.copy()
                            new_index[normalized_key]['cache_file'] = str(self.local_cache_dir / f"{normalized_key}_hypno.pkl")
                            # Remove old file if it's different
                            if str(existing_file) != str(current_file):
                                try:
                                    existing_file.unlink()
                                except Exception:
                                    pass
                        else:
                            # Remove current file if it's different
                            if str(existing_file) != str(current_file):
                                try:
                                    current_file.unlink()
                                except Exception:
                                    pass
                    elif current_file.exists():
                        new_index[normalized_key] = cache_info.copy()
                        new_index[normalized_key]['cache_file'] = str(self.local_cache_dir / f"{normalized_key}_hypno.pkl")
                    # else: existing_file exists, keep it
                else:
                    # New normalized key
                    cache_file_path = Path(cache_info.get('cache_file', ''))
                    normalized_file = self.local_cache_dir / f"{normalized_key}_hypno.pkl"
                    
                    # If cache file exists and has different name, rename it
                    if cache_file_path.exists() and cache_file_path != normalized_file:
                        try:
                            cache_file_path.rename(normalized_file)
                            migrated = True
                        except Exception as e:
                            self.logger.warning(f"Failed to rename cache file {cache_file_path} to {normalized_file}: {e}")
                            # If rename fails, check if normalized file already exists
                            if normalized_file.exists():
                                # Keep the normalized file, remove the old one
                                try:
                                    cache_file_path.unlink()
                                except Exception:
                                    pass
                    
                    # Update cache info
                    new_cache_info = cache_info.copy()
                    new_cache_info['cache_file'] = str(normalized_file)
                    new_cache_info['date'] = self._parse_date(date)  # Normalize date in info too
                    new_index[normalized_key] = new_cache_info
                    
                    if old_key != normalized_key:
                        migrated = True
        
        if migrated:
            self.cache_index = new_index
            self._save_cache_index()
            self.logger.info("Migrated cache index to normalized date format")
        
        # Clean up any remaining duplicate cache files
        self._cleanup_duplicate_cache_files()
    
    def _cleanup_duplicate_cache_files(self):
        """Remove duplicate cache files with different date formats."""
        if not self.local_cache_dir.exists():
            return
        
        # Find all cache files
        cache_files = list(self.local_cache_dir.glob('*_hypno.pkl'))
        
        # Group by rat_id and normalized date
        file_groups = {}
        for cache_file in cache_files:
            name = cache_file.stem  # e.g., "R2_2022-11-05_hypno" or "R2_2022_11_05_hypno"
            if '_hypno' in name:
                base_name = name.replace('_hypno', '')
                if '_' in base_name:
                    parts = base_name.split('_', 1)
                    if len(parts) == 2:
                        rat_id = parts[0]
                        date_part = parts[1]
                        normalized_date = self._parse_date(date_part)
                        normalized_key = f"{rat_id}_{normalized_date}"
                        
                        if normalized_key not in file_groups:
                            file_groups[normalized_key] = []
                        file_groups[normalized_key].append(cache_file)
        
        # For each group, keep only the file with normalized date format (underscores)
        removed_count = 0
        for normalized_key, files in file_groups.items():
            if len(files) > 1:
                # Find the file with normalized format (underscores, no dashes)
                normalized_file = None
                other_files = []
                
                for f in files:
                    # Check if this file matches the normalized key format (has underscores, no dashes in date part)
                    file_name = f.stem.replace('_hypno', '')
                    if '_' in file_name:
                        parts = file_name.split('_', 1)
                        if len(parts) == 2:
                            date_part = parts[1]
                            # Normalized format has underscores, not dashes
                            if '-' not in date_part and date_part == normalized_key.split('_', 1)[1]:
                                normalized_file = f
                            else:
                                other_files.append(f)
                    else:
                        other_files.append(f)
                
                # Keep normalized file, remove others
                if normalized_file:
                    for other_file in other_files:
                        try:
                            other_file.unlink()
                            removed_count += 1
                        except Exception as e:
                            self.logger.warning(f"Failed to remove duplicate cache file {other_file}: {e}")
                elif other_files:
                    # If no normalized file found, keep the first one and remove rest
                    # This shouldn't happen if migration worked, but handle it anyway
                    for other_file in other_files[1:]:
                        try:
                            other_file.unlink()
                            removed_count += 1
                        except Exception as e:
                            self.logger.warning(f"Failed to remove duplicate cache file {other_file}: {e}")
        
        if removed_count > 0:
            self.logger.info(f"Cleaned up {removed_count} duplicate cache files")
            
    def _generate_cache_key(self, rat_id: str, date: str) -> str:
        """Generate a unique cache key for rat and date."""
        # Normalize date format to ensure consistent cache keys
        date_normalized = self._parse_date(date)
        return f"{rat_id}_{date_normalized}"
        
    def _parse_date(self, date_str: str) -> str:
        """Parse date string to YYYY_MM_DD format."""
        return normalize_date_to_yyyymmdd(date_str)
            
    def _get_rat_directories(self) -> List[str]:
        """Get list of available rat directories."""
        rat_dirs = []
        if self.local_data_root.exists():
            for item in self.local_data_root.iterdir():
                if item.is_dir() and item.name.startswith('R'):
                    rat_dirs.append(item.name)
        return sorted(rat_dirs)
        
    def _get_available_dates(self, rat_id: str, source: str = 'local') -> List[str]:
        """Get list of available dates for a specific rat."""
        dates = []
        
        if source == 'local':
            # Check all date directories directly under /rat/
            if self.local_data_root.exists():
                for item in self.local_data_root.iterdir():
                    if item.is_dir() and len(item.name) == 10 and '_' in item.name:
                        # Check if this date has data for the specific rat
                        hypno_file = item / f"{rat_id}_hypno.pickle"
                        if hypno_file.exists():
                            dates.append(item.name)
        elif source == 's3' and self._get_s3_client():
            try:
                # List objects in S3 bucket for this rat
                response = self._get_s3_client().list_objects_v2(
                    Bucket=self.s3_temp_bucket,
                    Prefix=""  # List all objects
                )
                if 'Contents' in response:
                    for obj in response['Contents']:
                        key = obj['Key']
                        # Check if this is a hypnogram file for the specific rat
                        # S3 structure: date/rat/rat_hypno.pickle
                        if key.endswith(f"{rat_id}_hypno.pickle") and key.count('/') >= 2:
                            date_part = key.split('/')[0]
                            if len(date_part) == 10 and '_' in date_part:
                                dates.append(date_part)
            except Exception as e:
                self.logger.error(f"Failed to list S3 dates for {rat_id}: {e}")
                
        return sorted(dates)
        
    def _load_hypnogram_from_local(self, rat_id: str, date: str) -> Optional[np.ndarray]:
        """Load hypnogram from local/network drive."""
        try:
            date_parsed = self._parse_date(date)
            # Correct path structure: /rat/date/R2_hypno.pickle
            hypno_file = self.local_data_root / date_parsed / f"{rat_id}_hypno.pickle"
            
            if hypno_file.exists():
                with open(hypno_file, 'rb') as f:
                    hypnogram = pickle.load(f)
                return hypnogram
            else:
                # Only show this warning once per unique file to avoid spam
                warning_key = f"file_not_found_{hypno_file}"
                if warning_key not in self._shown_warnings:
                    self.logger.warning(f"Hypnogram file not found: {hypno_file}")
                    self._shown_warnings.add(warning_key)
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to load hypnogram from local: {e}")
            return None
            
    def _load_hypnogram_from_s3(self, rat_id: str, date: str) -> Optional[np.ndarray]:
        """Load hypnogram from S3 storage."""
        client = self._get_s3_client()
        if client is None:
            self.logger.error("S3 client not initialized")
            return None
            
        try:
            date_parsed = self._parse_date(date)
            # Correct S3 key structure: date/rat/rat_hypno.pickle
            s3_key = f"{date_parsed}/{rat_id}/{rat_id}_hypno.pickle"
            
            # Check if file already exists in temp bucket before downloading
            try:
                client.head_object(Bucket=self.s3_temp_bucket, Key=s3_key)
                # self.logger.info(f"File exists in S3 temp bucket: {s3_key}")  # Debug: uncomment if needed
                pass
            except client.exceptions.NoSuchKey:
                # self.logger.warning(f"File not found in S3 temp bucket: {s3_key}")  # Debug: uncomment if needed
                return None
            except Exception as e:
                # self.logger.error(f"Error checking S3 temp bucket for {s3_key}: {e}")  # Debug: uncomment if needed
                return None
            
            # Download to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pickle') as tmp_file:
                client.download_file(self.s3_temp_bucket, s3_key, tmp_file.name)
                
                # Load hypnogram
                with open(tmp_file.name, 'rb') as f:
                    hypnogram = pickle.load(f)
                    
                # Clean up temporary file
                os.unlink(tmp_file.name)
                
                return hypnogram
                
        except Exception as e:
            self.logger.error(f"Failed to load hypnogram from S3: {e}")
            return None
            
    def _load_eeg_data_from_local(self, rat_id: str, date: str) -> Optional[Tuple[np.ndarray, Dict]]:
        """Load EEG data from local/network drive."""
        try:
            date_parsed = self._parse_date(date)
            date_dir = self.local_data_root / rat_id / date_parsed
            
            # Find .dat file
            dat_file = None
            for file in date_dir.glob(f"{rat_id}_*.dat"):
                dat_file = file
                break
                
            if not dat_file:
                self.logger.warning(f"EEG data file not found for {rat_id} on {date}")
                return None
                
            # Load .inf file for channel information
            inf_file = date_dir / f"{rat_id}.inf"
            if not inf_file.exists():
                self.logger.warning(f"Info file not found: {inf_file}")
                return None
                
            with open(inf_file, 'r') as f:
                info_data = eval(f.read())
                
            # Load EEG data
            eeg_data = np.fromfile(dat_file, dtype=np.float32)
            
            # Reshape data based on channel count
            n_channels = info_data['electrodes']['count']
            n_samples = len(eeg_data) // n_channels
            eeg_data = eeg_data[:n_samples * n_channels].reshape(n_samples, n_channels)
            
            # Get channel names
            channel_names = info_data['electrodes']['names']
            
            # Select target channels
            target_indices = []
            for channel in self.target_channels:
                if channel in channel_names:
                    target_indices.append(channel_names.index(channel))
                else:
                    self.logger.warning(f"Channel {channel} not found in {channel_names}")
                    
            if not target_indices:
                self.logger.error(f"No target channels found in {channel_names}")
                return None
                
            # Extract target channels
            eeg_data = eeg_data[:, target_indices]
            
            return eeg_data, {
                'channel_names': [channel_names[i] for i in target_indices],
                'sampling_rate': info_data['sampling_rate'],
                'n_channels': len(target_indices),
                'n_samples': n_samples
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load EEG data from local: {e}")
            return None
            
    def _load_eeg_data_from_s3(self, rat_id: str, date: str) -> Optional[Tuple[np.ndarray, Dict]]:
        """Load EEG data from S3 storage."""
        client = self._get_s3_client()
        if client is None:
            self.logger.error("S3 client not initialized")
            return None
            
        try:
            date_parsed = self._parse_date(date)
            
            # Download .inf file
            inf_key = f"{rat_id}/{date_parsed}/{rat_id}.inf"
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.inf') as tmp_inf:
                client.download_file(self.s3_rat_bucket, inf_key, tmp_inf.name)
                
                with open(tmp_inf.name, 'r') as f:
                    info_data = eval(f.read())
                    
                os.unlink(tmp_inf.name)
                
            # Download .dat file
            dat_key = f"{rat_id}/{date_parsed}/{rat_id}_0.dat"
            with tempfile.NamedTemporaryFile(delete=False, suffix='.dat') as tmp_dat:
                client.download_file(self.s3_rat_bucket, dat_key, tmp_dat.name)
                
                # Load EEG data
                eeg_data = np.fromfile(tmp_dat.name, dtype=np.float32)
                os.unlink(tmp_dat.name)
                
            # Process data same as local
            n_channels = info_data['electrodes']['count']
            n_samples = len(eeg_data) // n_channels
            eeg_data = eeg_data[:n_samples * n_channels].reshape(n_samples, n_channels)
            
            channel_names = info_data['electrodes']['names']
            target_indices = []
            for channel in self.target_channels:
                if channel in channel_names:
                    target_indices.append(channel_names.index(channel))
                    
            if not target_indices:
                self.logger.error(f"No target channels found in {channel_names}")
                return None
                
            eeg_data = eeg_data[:, target_indices]
            
            return eeg_data, {
                'channel_names': [channel_names[i] for i in target_indices],
                'sampling_rate': info_data['sampling_rate'],
                'n_channels': len(target_indices),
                'n_samples': n_samples
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load EEG data from S3: {e}")
            return None
            
    def check_s3_reachable(self, timeout_s: float = 3.0) -> bool:
        """
        Quick connectivity test to the S3 endpoint.

        Caches the result so subsequent calls are instant.  Returns False
        (and marks S3 as unreachable for this instance) when the endpoint
        cannot be contacted within *timeout_s* seconds.
        """
        if self._s3_reachable is not None:
            return self._s3_reachable

        client = self._get_s3_client()
        if client is None:
            self._s3_reachable = False
            return False

        import socket, urllib.parse
        try:
            parsed = urllib.parse.urlparse(self.s3_config.get('endpoint_url', ''))
            host = parsed.hostname or ''
            port = parsed.port or 80
            sock = socket.create_connection((host, port), timeout=timeout_s)
            sock.close()
            self._s3_reachable = True
        except Exception:
            self._s3_reachable = False
        return self._s3_reachable

    def is_known_missing(self, rat_id: str, date: str) -> bool:
        """Return True if (rat_id, date) was already determined to be missing."""
        return (rat_id, date) in self._missing_negative_cache

    def mark_missing(self, rat_id: str, date: str) -> None:
        """Add (rat_id, date) to the negative cache."""
        self._missing_negative_cache.add((rat_id, date))

    def _check_s3_temp_bucket_exists(self, rat_id: str, date: str) -> bool:
        """
        Check if hypnogram file exists in S3 temp bucket without downloading.
        
        Parameters
        ----------
        rat_id : str
            Rat identifier
        date : str
            Date in YYYY_MM_DD format
            
        Returns
        -------
        bool
            True if file exists in S3 temp bucket, False otherwise
        """
        # Fast-path: if S3 is known unreachable, don't bother
        if self._s3_reachable is False:
            return False

        # #region agent log
        if not hasattr(self, '_dbg_s3_check_count'): self._dbg_s3_check_count = 0
        self._dbg_s3_check_count += 1
        _dbg_should_log = self._dbg_s3_check_count <= 30
        # #endregion
        client = self._get_s3_client()
        if client is None:
            # #region agent log
            if _dbg_should_log:
                import time as _t, json as _j, os as _os
                _logpath = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))), '.cursor', 'debug.log')
                try:
                    with open(_logpath, 'a') as _f:
                        _f.write(_j.dumps({"hypothesisId":"H4","location":"cache_manager.py:_check_s3:no_client","message":"s3_client_is_none","data":{"rat_id":rat_id,"date":date,"call_num":self._dbg_s3_check_count},"timestamp":int(_t.time()*1000)}) + '\n')
                except Exception:
                    pass
            # #endregion
            return False
            
        try:
            # #region agent log
            import time as _t; _s3t0 = _t.time()
            # #endregion
            date_parsed = self._parse_date(date)
            s3_key = f"{date_parsed}/{rat_id}/{rat_id}_hypno.pickle"
            client.head_object(Bucket=self.s3_temp_bucket, Key=s3_key)
            # #region agent log
            if _dbg_should_log:
                _s3elapsed = _t.time() - _s3t0
                import json as _j, os as _os
                _logpath = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))), '.cursor', 'debug.log')
                try:
                    with open(_logpath, 'a') as _f:
                        _f.write(_j.dumps({"hypothesisId":"H1","location":"cache_manager.py:_check_s3:found","message":"s3_check_found","data":{"rat_id":rat_id,"date":date,"elapsed_s":round(_s3elapsed,4),"call_num":self._dbg_s3_check_count},"timestamp":int(_t.time()*1000)}) + '\n')
                except Exception:
                    pass
            # #endregion
            return True
        except client.exceptions.NoSuchKey:
            # #region agent log
            if _dbg_should_log:
                _s3elapsed = _t.time() - _s3t0
                import json as _j, os as _os
                _logpath = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))), '.cursor', 'debug.log')
                try:
                    with open(_logpath, 'a') as _f:
                        _f.write(_j.dumps({"hypothesisId":"H1","location":"cache_manager.py:_check_s3:not_found","message":"s3_check_not_found","data":{"rat_id":rat_id,"date":date,"elapsed_s":round(_s3elapsed,4),"call_num":self._dbg_s3_check_count},"timestamp":int(_t.time()*1000)}) + '\n')
                except Exception:
                    pass
            # #endregion
            return False
        except Exception as e:
            # #region agent log
            if _dbg_should_log:
                _s3elapsed = _t.time() - _s3t0
                import json as _j, os as _os
                _logpath = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))), '.cursor', 'debug.log')
                try:
                    with open(_logpath, 'a') as _f:
                        _f.write(_j.dumps({"hypothesisId":"H1_H4","location":"cache_manager.py:_check_s3:error","message":"s3_check_error","data":{"rat_id":rat_id,"date":date,"elapsed_s":round(_s3elapsed,4),"error":str(e)[:200],"call_num":self._dbg_s3_check_count},"timestamp":int(_t.time()*1000)}) + '\n')
                except Exception:
                    pass
            # #endregion
            # self.logger.error(f"Error checking S3 temp bucket for {rat_id}_{date}: {e}")  # Debug: uncomment if needed
            return False

    def cache_hypnogram(self, rat_id: str, date: str, source: str = 'local') -> bool:
        """
        Cache hypnogram file for a specific rat and date.
        
        Parameters
        ----------
        rat_id : str
            Rat identifier (e.g., 'R1', 'R2', etc.)
        date : str
            Date in YYYY_MM_DD format
        source : str, default='local'
            Source to load from ('local' or 's3')
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            cache_key = self._generate_cache_key(rat_id, date)
            
            # Check if already cached - validate the cache file exists and is readable
            if cache_key in self.cache_index:
                cache_info = self.cache_index[cache_key]
                cache_file = Path(cache_info['cache_file'])
                # Validate cache file exists and is readable
                if cache_file.exists():
                    try:
                        with open(cache_file, 'rb') as f:
                            test_hypno = pickle.load(f)
                        if test_hypno is not None:
                            # Cache is valid
                            return True
                        else:
                            # Cache file exists but contains None - remove it
                            self.logger.warning(f"Cache file contains None for {cache_key}, removing")
                            cache_file.unlink()
                            del self.cache_index[cache_key]
                            self._save_cache_index()
                    except Exception as e:
                        # Cache file is corrupted - remove it
                        self.logger.warning(f"Cache file corrupted for {cache_key}: {e}, removing")
                        try:
                            cache_file.unlink()
                        except Exception:
                            pass
                        del self.cache_index[cache_key]
                        self._save_cache_index()
                else:
                    # Cache file doesn't exist - remove from index
                    del self.cache_index[cache_key]
                    self._save_cache_index()
                
            # Load hypnogram from source
            if source == 'local':
                hypnogram = self._load_hypnogram_from_local(rat_id, date)
            elif source == 's3':
                # Check if file exists in S3 temp bucket before attempting download
                if not self._check_s3_temp_bucket_exists(rat_id, date):
                    # self.logger.warning(f"File not found in S3 temp bucket: {rat_id}_{date}")  # Debug: uncomment if needed
                    return False
                hypnogram = self._load_hypnogram_from_s3(rat_id, date)
            else:
                self.logger.error(f"Unknown source: {source}")
                return False
                
            if hypnogram is None:
                # self.logger.warning(f"Failed to load hypnogram for {rat_id} on {date}")  # Debug: uncomment if needed
                return False
            
            # Validate hypnogram data before attempting to cache
            try:
                if isinstance(hypnogram, list):
                    if len(hypnogram) == 0:
                        self.logger.warning(f"Hypnogram is empty list for {rat_id} on {date}")
                        return False
                    # Check first element
                    if hypnogram[0] is None:
                        self.logger.warning(f"Hypnogram first element is None for {rat_id} on {date}")
                        return False
                elif isinstance(hypnogram, np.ndarray):
                    if hypnogram.size == 0:
                        self.logger.warning(f"Hypnogram array is empty for {rat_id} on {date}")
                        return False
                else:
                    self.logger.warning(f"Hypnogram has unexpected type {type(hypnogram)} for {rat_id} on {date}")
                    return False
            except Exception as validation_error:
                self.logger.error(f"Error validating hypnogram for {rat_id} on {date}: {validation_error}")
                return False
                
            # Save to cache using atomic write to prevent corruption
            # Ensure cache directory exists (defensive check)
            try:
                self.local_cache_dir.mkdir(parents=True, exist_ok=True)
            except Exception as dir_error:
                self.logger.error(f"Failed to create cache directory {self.local_cache_dir}: {dir_error}")
                return False
            
            # Use absolute paths to avoid issues with relative paths in parallel execution
            # Add process ID to temp file name to avoid race conditions
            import os as os_module
            temp_suffix = f'.pkl.tmp.{os_module.getpid()}'
            cache_file = self.local_cache_dir.resolve() / f"{cache_key}_hypno.pkl"
            temp_file = cache_file.parent / f"{cache_key}_hypno{temp_suffix}"
            
            try:
                # Write to temporary file first
                try:
                    with open(temp_file, 'wb') as f:
                        pickle.dump(hypnogram, f, protocol=pickle.HIGHEST_PROTOCOL)
                        f.flush()
                        os.fsync(f.fileno())  # Ensure data is written to disk
                except Exception as write_error:
                    self.logger.error(f"Failed to write temp file {temp_file} for {rat_id} on {date}: {write_error}")
                    # Clean up temp file if it exists
                    try:
                        if temp_file.exists():
                            temp_file.unlink()
                    except Exception:
                        pass
                    raise write_error
                
                # Verify temp file was written successfully
                if not temp_file.exists():
                    raise ValueError(f"Temporary cache file was not created: {temp_file}")
                temp_file_size = temp_file.stat().st_size
                if temp_file_size == 0:
                    raise ValueError(f"Temporary cache file is empty: {temp_file}")
                
                # Atomic rename using os.replace (more reliable than Path.replace)
                # Use absolute paths for os.replace
                temp_file_str = str(temp_file.resolve())
                cache_file_str = str(cache_file.resolve())
                os.replace(temp_file_str, cache_file_str)
                
                # Verify the final file was written correctly
                if not cache_file.exists():
                    raise ValueError(f"Cache file verification failed: file is missing after rename: {cache_file}")
                cache_file_size = cache_file.stat().st_size
                if cache_file_size == 0:
                    raise ValueError(f"Cache file verification failed: file is empty: {cache_file}")
                if cache_file_size != temp_file_size:
                    self.logger.warning(f"Cache file size mismatch: temp={temp_file_size}, final={cache_file_size}")
                
                # Update cache index
                self.cache_index[cache_key] = {
                    'rat_id': rat_id,
                    'date': date,
                    'source': source,
                    'cache_file': str(cache_file),
                    'cached_at': datetime.now().isoformat(),
                    'hypnogram_shape': hypnogram.shape if hasattr(hypnogram, 'shape') else len(hypnogram)
                }
                self._save_cache_index()
            except Exception as e:
                error_msg = f"Failed to cache hypnogram for {rat_id} on {date}: {type(e).__name__} - {e}"
                self.logger.error(error_msg)
                
                # Clean up temp file if write failed
                try:
                    if temp_file.exists():
                        temp_file.unlink()
                        self.logger.debug(f"Cleaned up temp file: {temp_file}")
                except Exception as cleanup_error:
                    self.logger.warning(f"Failed to clean up temp file {temp_file}: {cleanup_error}")
                
                # Remove cache file if it exists but is corrupted
                try:
                    if cache_file.exists():
                        if cache_file.stat().st_size == 0:
                            cache_file.unlink()
                            self.logger.debug(f"Removed empty cache file: {cache_file}")
                except Exception as cleanup_error:
                    self.logger.warning(f"Failed to clean up corrupted cache file {cache_file}: {cleanup_error}")
                
                # Don't raise - return False so the outer handler can log it
                return False
            
            # self.logger.info(f"Successfully cached hypnogram: {cache_key}")  # Debug: uncomment if needed
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cache hypnogram: {e}")
            return False

    def cache_hypnogram_from_data(self, rat_id: str, date: str, hypnogram: Union[np.ndarray, list]) -> bool:
        """
        Cache a hypnogram that was computed in-memory (e.g. from .dat files).
        Saves directly to local cache without loading from file.

        Parameters
        ----------
        rat_id : str
            Rat identifier (e.g., 'R1', 'R2', etc.)
        date : str
            Date in YYYY_MM_DD format
        hypnogram : np.ndarray or list
            Hypnogram data to cache (computed in-memory)

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            cache_key = self._generate_cache_key(rat_id, date)

            # Check if already cached
            if cache_key in self.cache_index:
                cache_info = self.cache_index[cache_key]
                cache_file = Path(cache_info['cache_file'])
                if cache_file.exists():
                    try:
                        with open(cache_file, 'rb') as f:
                            test_hypno = pickle.load(f)
                        if test_hypno is not None:
                            return True
                    except Exception:
                        pass

            # Validate hypnogram
            if hypnogram is None:
                self.logger.warning(f"Hypnogram is None for {rat_id} on {date}")
                return False
            if isinstance(hypnogram, list) and len(hypnogram) == 0:
                self.logger.warning(f"Hypnogram is empty list for {rat_id} on {date}")
                return False
            if isinstance(hypnogram, np.ndarray) and hypnogram.size == 0:
                self.logger.warning(f"Hypnogram array is empty for {rat_id} on {date}")
                return False

            self.local_cache_dir.mkdir(parents=True, exist_ok=True)
            import os as os_module
            temp_suffix = f'.pkl.tmp.{os_module.getpid()}'
            cache_file = self.local_cache_dir.resolve() / f"{cache_key}_hypno.pkl"
            temp_file = cache_file.parent / f"{cache_key}_hypno{temp_suffix}"

            try:
                with open(temp_file, 'wb') as f:
                    pickle.dump(hypnogram, f, protocol=pickle.HIGHEST_PROTOCOL)
                    f.flush()
                    os.fsync(f.fileno())
                if not temp_file.exists() or temp_file.stat().st_size == 0:
                    raise ValueError("Temp file write failed")
                os.replace(str(temp_file.resolve()), str(cache_file.resolve()))
                self.cache_index[cache_key] = {
                    'rat_id': rat_id,
                    'date': date,
                    'source': 'computed',
                    'cache_file': str(cache_file),
                    'cached_at': datetime.now().isoformat(),
                    'hypnogram_shape': hypnogram.shape if hasattr(hypnogram, 'shape') else len(hypnogram)
                }
                self._save_cache_index()
                return True
            except Exception as e:
                self.logger.error(f"Failed to cache computed hypnogram for {rat_id} on {date}: {e}")
                try:
                    if temp_file.exists():
                        temp_file.unlink()
                except Exception:
                    pass
                return False
        except Exception as e:
            self.logger.error(f"Failed to cache hypnogram from data: {e}")
            return False

    def cache_multiple_hypnograms(self, rat_dates: List[Tuple[str, str]], source: str = 'local') -> Dict[str, bool]:
        """
        Cache multiple hypnogram files.
        
        Parameters
        ----------
        rat_dates : list of tuples
            List of (rat_id, date) tuples
        source : str, default='local'
            Source to load from ('local' or 's3')
            
        Returns
        -------
        dict
            Dictionary mapping cache keys to success status
        """
        results = {}
        
        for rat_id, date in rat_dates:
            cache_key = self._generate_cache_key(rat_id, date)
            results[cache_key] = self.cache_hypnogram(rat_id, date, source)
            
        return results
        
    def get_cached_hypnogram(self, rat_id: str, date: str) -> Optional[np.ndarray]:
        """
        Get cached hypnogram for a specific rat and date.
        
        Parameters
        ----------
        rat_id : str
            Rat identifier
        date : str
            Date in YYYY_MM_DD format
            
        Returns
        -------
        np.ndarray or None
            Cached hypnogram if found, None otherwise
        """
        try:
            cache_key = self._generate_cache_key(rat_id, date)
            
            if cache_key not in self.cache_index:
                # Only show this warning once per unique cache key to avoid spam
                if cache_key not in self._shown_warnings:
                    self.logger.warning(f"Hypnogram not cached: {cache_key}")
                    self._shown_warnings.add(cache_key)
                return None
                
            cache_info = self.cache_index[cache_key]
            cache_file = Path(cache_info['cache_file'])
            
            if not cache_file.exists():
                self.logger.warning(f"Cache file not found: {cache_file}")
                # Remove from index
                del self.cache_index[cache_key]
                self._save_cache_index()
                return None
                
            # Validate file size before attempting to load (quick check for truncated files)
            file_size = cache_file.stat().st_size
            if file_size == 0:
                raise ValueError("Cache file is empty (0 bytes)")
            
            with open(cache_file, 'rb') as f:
                hypnogram = pickle.load(f)
            
            # Validate that we got valid data
            if hypnogram is None:
                raise ValueError("Cached hypnogram is None")
            
            # Additional validation: check if it's a valid numpy array or list
            if not isinstance(hypnogram, (np.ndarray, list)):
                raise ValueError(f"Invalid hypnogram type: {type(hypnogram)}")
                
            return hypnogram
            
        except (EOFError, ValueError, pickle.UnpicklingError, OSError) as e:
            # These are corruption-related errors - clean up and try to reload from source
            error_type = type(e).__name__
            cache_key = self._generate_cache_key(rat_id, date)
            
            if "truncated" in str(e) or "Ran out of input" in str(e) or isinstance(e, EOFError) or "empty" in str(e).lower():
                self.logger.warning(f"Corrupted/empty cache file detected for {rat_id}_{date}: {error_type} - {e}. Removing and attempting to reload from source.")
            else:
                self.logger.error(f"Failed to get cached hypnogram for {rat_id}_{date}: {error_type} - {e}")
            
            # Remove corrupted cache entry from index
            if cache_key in self.cache_index:
                cache_info = self.cache_index[cache_key]
                cache_file = Path(cache_info['cache_file'])
                # Try to remove corrupted cache file
                try:
                    if cache_file.exists():
                        cache_file.unlink()
                        self.logger.debug(f"Removed corrupted cache file: {cache_file}")
                except Exception as cleanup_error:
                    self.logger.warning(f"Failed to remove corrupted cache file {cache_file}: {cleanup_error}")
                # Remove from index
                del self.cache_index[cache_key]
                self._save_cache_index()
            
            # Try to automatically reload from source
            self.logger.info(f"Attempting to reload hypnogram for {rat_id}_{date} from source...")
            try:
                # Try local first
                success = self.cache_hypnogram(rat_id, date, source='local')
                if success:
                    # Verify it was actually cached correctly
                    hypnogram = self.get_cached_hypnogram(rat_id, date)
                    if hypnogram is not None:
                        self.logger.info(f"Successfully reloaded and cached hypnogram for {rat_id}_{date}")
                        return hypnogram
                
                # Try S3 fallback
                if self._check_s3_temp_bucket_exists(rat_id, date):
                    success = self.cache_hypnogram(rat_id, date, source='s3')
                    if success:
                        hypnogram = self.get_cached_hypnogram(rat_id, date)
                        if hypnogram is not None:
                            self.logger.info(f"Successfully reloaded and cached hypnogram for {rat_id}_{date} from S3")
                            return hypnogram
            except Exception as reload_error:
                self.logger.warning(f"Failed to reload hypnogram for {rat_id}_{date}: {reload_error}")
            
            return None
        except Exception as e:
            # Other unexpected errors
            self.logger.error(f"Unexpected error getting cached hypnogram for {rat_id}_{date}: {type(e).__name__} - {e}")
            return None
            
    def get_available_data(self, source: str = 'local') -> Dict[str, List[str]]:
        """
        Get available data for all rats.
        
        Parameters
        ----------
        source : str, default='local'
            Source to check ('local' or 's3')
            
        Returns
        -------
        dict
            Dictionary mapping rat_id to list of available dates
        """
        available_data = {}
        
        if source == 'local':
            # Find all rats by looking for hypnogram files
            if self.local_data_root.exists():
                rats = set()
                for date_dir in self.local_data_root.iterdir():
                    if date_dir.is_dir() and len(date_dir.name) == 10 and '_' in date_dir.name:
                        # Look for hypnogram files in this date directory
                        for file in date_dir.iterdir():
                            if file.name.endswith('_hypno.pickle') and file.name.startswith('R'):
                                rat_id = file.name.replace('_hypno.pickle', '')
                                rats.add(rat_id)
                
                # Get dates for each rat
                for rat_id in rats:
                    dates = self._get_available_dates(rat_id, 'local')
                    if dates:
                        available_data[rat_id] = dates
        elif source == 's3' and self._get_s3_client():
            # Get all rats from S3
            try:
                response = self._get_s3_client().list_objects_v2(Bucket=self.s3_rat_bucket)
                if 'Contents' in response:
                    rats = set()
                    for obj in response['Contents']:
                        key = obj['Key']
                        if '/' in key:
                            rat_id = key.split('/')[0]
                            if rat_id.startswith('R'):
                                rats.add(rat_id)
                    
                    for rat_id in rats:
                        dates = self._get_available_dates(rat_id, 's3')
                        if dates:
                            available_data[rat_id] = dates
            except Exception as e:
                self.logger.error(f"Failed to get available data from S3: {e}")
                
        return available_data
        
    def clear_cache(self):
        """Clear all cached files."""
        try:
            # Remove all cache files
            for cache_file in self.local_cache_dir.glob('*_hypno.pkl'):
                cache_file.unlink()
                
            # Clear index
            self.cache_index.clear()
            self._save_cache_index()
            
            self.logger.info("Cache cleared successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
            
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the cache."""
        total_files = len(self.cache_index)
        total_size = sum(f.stat().st_size for f in self.local_cache_dir.glob('*_hypno.pkl'))
        
        return {
            'total_files': total_files,
            'total_size_bytes': total_size,
            'cache_dir': str(self.local_cache_dir),
            'local_data_root': str(self.local_data_root),
            's3_configured': self.s3_config is not None,
            'target_channels': self.target_channels,
            'sampling_rate': self.sampling_rate
        }
        
    def get_cache_status(self) -> Dict[str, Any]:
        """Get detailed cache status."""
        status = {
            'cached_files': list(self.cache_index.keys()),
            'cache_info': self.get_cache_info(),
            'available_local': self.get_available_data('local'),
            'available_s3': self.get_available_data('s3') if self._get_s3_client() else {}
        }
        
        return status

    # -----------------------------------------------------------------------------
    #  Internal helpers for safe (de)serialization and shared S3 clients
    # -----------------------------------------------------------------------------
    def _get_s3_client(self):
        """Lazily create or fetch a boto3 client for the current configuration."""
        if self._s3_client is not None:
            return self._s3_client
        if not self.s3_config:
            return None
        
        # Ensure class attribute exists (important for cloned instances during grid search)
        if not hasattr(HypnogramCacheManagerYt, "_CLIENT_CACHE"):
            HypnogramCacheManagerYt._CLIENT_CACHE = {}
        
        cfg_key = frozenset(self.s3_config.items())
        if cfg_key in HypnogramCacheManagerYt._CLIENT_CACHE:
            self._s3_client = HypnogramCacheManagerYt._CLIENT_CACHE[cfg_key]
        else:
            # Separate service_name from other kwargs for boto3.client
            cfg = dict(self.s3_config)
            service_name = cfg.pop('service_name', 's3')
            self._s3_client = boto3.client(service_name, **cfg)
            HypnogramCacheManagerYt._CLIENT_CACHE[cfg_key] = self._s3_client
        return self._s3_client

    # Make the class picklable by excluding the boto3 client
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_s3_client'] = None  # remove un-picklable client
        # Preserve negative cache and S3 reachability across pickle (joblib)
        # These are plain Python sets / booleans, so they pickle fine.
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._s3_client = None  # will be recreated lazily
        # Ensure negative cache exists (backward compat with old pickles)
        if not hasattr(self, '_missing_negative_cache'):
            self._missing_negative_cache = set()
        if not hasattr(self, '_s3_reachable'):
            self._s3_reachable = None

    def precache_for_experiment(
        self,
        events: List[Dict],
        window_positions: List[int],
        window_days: int = 3,
        fixed_control_start_days: int = 9,
        control_window_days: Optional[int] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Pre-cache all hypnograms needed for experiments.

        Parameters
        ----------
        events : list of dict
            List of events with 'rat_id' and 'date' keys.
        window_positions : list of int
            List of window positions to test (e.g., [5, 4, 3, ..., -8]).
        window_days : int, default=3
            Number of days in each window.
        fixed_control_start_days : int, default=9
            Start day for fixed control window.
        control_window_days : int, optional
            Control window size in days. Defaults to window_days.
        progress_callback : callable, optional
            If provided, called with progress messages (str). When None, silent.

        Returns
        -------
        dict
            Keys: 'cached', 'missing', 'total', 'missing_list'.
        """
        def _log(msg: str) -> None:
            if progress_callback is not None:
                progress_callback(msg)

        control_window_days = control_window_days if control_window_days is not None else window_days
        dates_needed = set()

        for event in events:
            rat_id = event['rat_id']
            event_date_str = event['date']
            try:
                event_date = parse_event_date(event_date_str)
            except ValueError:
                _log(f"Warning: Could not parse date {event_date_str} for {rat_id}")
                continue

            for day_offset in range(
                fixed_control_start_days,
                fixed_control_start_days - control_window_days,
                -1,
            ):
                date_needed = event_date - timedelta(days=day_offset)
                dates_needed.add((rat_id, date_needed.strftime('%Y_%m_%d')))

            for window_pos in window_positions:
                abs_step = max(1, window_pos)
                immediate_start_offset = abs_step + window_days - 1
                for day_offset in range(
                    immediate_start_offset,
                    immediate_start_offset - window_days,
                    -1,
                ):
                    date_needed = event_date - timedelta(days=day_offset)
                    dates_needed.add((rat_id, date_needed.strftime('%Y_%m_%d')))

        _log(f"Total unique hypnograms needed: {len(dates_needed)}")

        # --- Check S3 reachability once upfront (fast 3 s timeout) ---
        import time as _precache_time
        s3_reachable = self.check_s3_reachable(timeout_s=3.0)
        if s3_reachable:
            _log("S3 endpoint is reachable – will try S3 fallback for missing local data.")
        else:
            _log("S3 endpoint is UNREACHABLE – skipping all S3 lookups (saves minutes of timeouts).")

        _log("Caching hypnograms...\n")
        _precache_t0 = _precache_time.time()

        cached_count = 0
        missing_hypnograms = []

        for rat_id, date_str in sorted(dates_needed):
            cache_key = self._generate_cache_key(rat_id, date_str)
            if cache_key in self.cache_index:
                cache_info = self.cache_index[cache_key]
                cache_file = Path(cache_info['cache_file'])
                if cache_file.exists():
                    try:
                        file_size = cache_file.stat().st_size
                        if file_size > 0:
                            hypnogram = self.get_cached_hypnogram(rat_id, date_str)
                            if hypnogram is not None:
                                cached_count += 1
                                if cached_count % 10 == 0:
                                    _log(f"  Cached {cached_count}/{len(dates_needed)} hypnograms...")
                                continue
                            self.logger.warning(
                                f"Invalid cache file for {rat_id} {date_str}, re-caching..."
                            )
                        else:
                            self.logger.warning(
                                f"Empty cache file for {rat_id} {date_str}, re-caching..."
                            )
                        try:
                            cache_file.unlink()
                        except Exception:
                            pass
                        del self.cache_index[cache_key]
                        self._save_cache_index()
                    except Exception as e:
                        self.logger.warning(
                            f"Error validating cache file for {rat_id} {date_str}: {e}"
                        )
                        try:
                            if cache_file.exists():
                                cache_file.unlink()
                        except Exception:
                            pass
                        if cache_key in self.cache_index:
                            del self.cache_index[cache_key]
                            self._save_cache_index()
                else:
                    if cache_key in self.cache_index:
                        del self.cache_index[cache_key]
                        self._save_cache_index()

            success = self.cache_hypnogram(rat_id, date_str, source='local')
            if not success and self._check_s3_temp_bucket_exists(rat_id, date_str):
                success = self.cache_hypnogram(rat_id, date_str, source='s3')

            if success:
                hypnogram = self.get_cached_hypnogram(rat_id, date_str)
                if hypnogram is not None:
                    cached_count += 1
                    if cached_count % 10 == 0:
                        _log(f"  Cached {cached_count}/{len(dates_needed)} hypnograms...")
                else:
                    _log(f"  WARNING: {rat_id} {date_str} - Caching reported success but file validation failed")
            else:
                date_parsed = self._parse_date(date_str)
                expected_path = self.local_data_root / date_parsed / f"{rat_id}_hypno.pickle"
                if expected_path.exists():
                    try:
                        hypnogram = self._load_hypnogram_from_local(rat_id, date_str)
                        if hypnogram is not None:
                            cache_file = self.local_cache_dir / f"{cache_key}_hypno.pkl"
                            with open(cache_file, 'wb') as f:
                                pickle.dump(hypnogram, f)
                            self.cache_index[cache_key] = {
                                'rat_id': rat_id,
                                'date': date_str,
                                'source': 'local',
                                'cache_file': str(cache_file),
                                'cached_at': datetime.now().isoformat(),
                                'hypnogram_shape': hypnogram.shape if hasattr(hypnogram, 'shape') else len(hypnogram),
                            }
                            self._save_cache_index()
                            cached_count += 1
                            if cached_count % 10 == 0:
                                _log(f"  Cached {cached_count}/{len(dates_needed)} hypnograms...")
                            continue
                    except Exception as e:
                        self.logger.warning(f"File exists but load failed for {rat_id} {date_str}: {e}")
                missing_hypnograms.append((rat_id, date_str))
                # Populate negative cache so grid search skips these instantly
                self.mark_missing(rat_id, date_str)

        _precache_elapsed = _precache_time.time() - _precache_t0
        _log(f"\nCaching complete! (took {_precache_elapsed:.1f}s)")
        _log(f"  Successfully cached: {cached_count}/{len(dates_needed)}")
        _log(f"  Missing: {len(missing_hypnograms)}/{len(dates_needed)}")
        if missing_hypnograms:
            _log(f"  (Missing dates added to negative cache – grid search will skip them instantly)")

        if missing_hypnograms:
            _log("\n=== MISSING HYPNOGRAMS ===\n")
            _log(f"Total missing: {len(missing_hypnograms)}\n")
            missing_date_to_events = {}
            for event in events:
                rat_id = event['rat_id']
                event_date_str = event['date']
                try:
                    event_date = parse_event_date(event_date_str)
                except ValueError:
                    continue
                for day_offset in range(
                    fixed_control_start_days,
                    fixed_control_start_days - control_window_days,
                    -1,
                ):
                    date_needed = event_date - timedelta(days=day_offset)
                    date_key = (rat_id, date_needed.strftime('%Y_%m_%d'))
                    if date_key in missing_hypnograms:
                        if date_key not in missing_date_to_events:
                            missing_date_to_events[date_key] = []
                        missing_date_to_events[date_key].append({
                            'event': f"{rat_id} {event_date_str}",
                            'window_type': 'control',
                            'day_offset': day_offset,
                        })
                for window_pos in window_positions:
                    abs_step = max(1, window_pos)
                    immediate_start_offset = abs_step + window_days - 1
                    for day_offset in range(
                        immediate_start_offset,
                        immediate_start_offset - window_days,
                        -1,
                    ):
                        date_needed = event_date - timedelta(days=day_offset)
                        date_key = (rat_id, date_needed.strftime('%Y_%m_%d'))
                        if date_key in missing_hypnograms:
                            if date_key not in missing_date_to_events:
                                missing_date_to_events[date_key] = []
                            missing_date_to_events[date_key].append({
                                'event': f"{rat_id} {event_date_str}",
                                'window_type': 'immediate',
                                'window_pos': window_pos,
                                'day_offset': day_offset,
                            })
            missing_by_rat = {}
            for rat_id, date_str in missing_hypnograms:
                if rat_id not in missing_by_rat:
                    missing_by_rat[rat_id] = []
                missing_by_rat[rat_id].append(date_str)
            for rat_id in sorted(missing_by_rat.keys()):
                dates = sorted(missing_by_rat[rat_id])
                _log(f"  {rat_id}: {len(dates)} missing dates")
                for date_str in dates:
                    date_key = (rat_id, date_str)
                    events_needing = missing_date_to_events.get(date_key, [])
                    if events_needing:
                        event_list = ', '.join([e['event'] for e in events_needing[:3]])
                        if len(events_needing) > 3:
                            event_list += f" (+{len(events_needing)-3} more)"
                        _log(f"    - {date_str} (needed for: {event_list})")
                    else:
                        _log(f"    - {date_str}")
                _log("")
        else:
            _log("\nAll hypnograms successfully cached!\n")

        return {
            'cached': cached_count,
            'missing': len(missing_hypnograms),
            'total': len(dates_needed),
            'missing_list': missing_hypnograms,
        }

