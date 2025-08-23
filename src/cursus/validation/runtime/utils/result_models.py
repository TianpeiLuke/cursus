"""Result models for script testing."""

from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class ExecutionResult(BaseModel):
    """Result of script execution"""
    success: bool
    execution_time: float
    memory_usage: int  # MB
    result_data: Optional[Any] = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None

class TestResult(BaseModel):
    """Result of script functionality test"""
    script_name: str
    status: str  # PASS, FAIL, SKIP
    execution_time: float
    memory_usage: int
    error_message: Optional[str] = None
    recommendations: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    def is_successful(self) -> bool:
        """Check if test was successful"""
        return self.status == "PASS"
