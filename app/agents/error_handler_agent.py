import functools
import logging
import traceback
from fastapi import HTTPException
from fastapi.responses import JSONResponse

class ErrorHandlerAgent:
    """
    A centralized error handler for all agents.
    Logs error details and returns a consistent message format.
    """

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    def handle_error(self, error: Exception, agent_name: str = "UnknownAgent", method: str = "unknown_method"):
        """
        Logs error and raises an HTTPException.
        """
        error_trace = traceback.format_exc()
        self.logger.error(
            f"[{agent_name}.{method}] Error: {str(error)}\nTraceback:\n{error_trace}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"[{agent_name}] encountered an issue in `{method}`: {str(error)}"
        )


    def safe_execute(self, func, *args, agent_name: str = "UnknownAgent", method: str = "unknown_method", **kwargs):
        """
        Executes a synchronous function safely, catching exceptions and returning a structured error.
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # If already an HTTPException (possibly from an agent), return it as JSON
            if isinstance(e, HTTPException):
                detail = e.detail
                if isinstance(detail, dict) and "error" in detail:
                    # Already structured error detail
                    return JSONResponse(status_code=e.status_code, content=detail)
                else:
                    # Wrap non-structured detail into our error format
                    error_msg = detail if isinstance(detail, str) else str(detail)
                    return JSONResponse(status_code=e.status_code, content={"error": error_msg})
            # For any other exception, log and return standardized JSON error
            return self.handle_error(e)
            



    async def safe_execute_async(self, func, *args, agent_name: str = "UnknownAgent", method: str = "unknown_method", **kwargs):
        """
        Executes an async function safely, catching exceptions and returning a structured error.
        """
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if isinstance(e, HTTPException):
                detail = e.detail
                if isinstance(detail, dict) and "error" in detail:
                    return JSONResponse(status_code=e.status_code, content=detail)
                else:
                    error_msg = detail if isinstance(detail, str) else str(detail)
                    return JSONResponse(status_code=e.status_code, content={"error": error_msg})
            return self.handle_error(e)
        

    def handle_errors(self, agent_name="UnknownAgent", method="unknown_method"):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self.safe_execute(func, *args, agent_name=agent_name, method=method, **kwargs)
            return wrapper
        return decorator

    def handle_errors_async(self, agent_name="UnknownAgent", method="unknown_method"):
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                return await self.safe_execute_async(func, *args, agent_name=agent_name, method=method, **kwargs)
            return wrapper
        return decorator