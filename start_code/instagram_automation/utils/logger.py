"""Instagram Automation - Logging Utility"""

import os
from datetime import datetime
from typing import Optional


class Logger:
    """Simple logging utility for Instagram automation"""

    def __init__(self, log_file: str = "logs/automation.log"):
        self.log_file = log_file
        
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def log(self, message: str, level: str = "INFO"):
        """Log a message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
    
    def info(self, message: str):
        """Log info message"""
        self.log(message, "INFO")
        print(f"ℹ️  {message}")

    def warning(self, message: str):
        """Log warning message"""
        self.log(message, "WARNING")
        print(f"⚠️  {message}")

    def error(self, message: str):
        """Log error message"""
        self.log(message, "ERROR")
        print(f"❌ {message}")

    def success(self, message: str):
        """Log success message"""
        self.log(message, "SUCCESS")
        print(f"✅ {message}")

    def api_log(self, service: str, action: str, cost: float, details: str = ""):
        """Log API usage"""
        message = f"{service} - {action}: ${cost:.4f} - {details}"
        self.log(message, "API")
