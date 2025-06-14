# app/agents/error_handler_agent.py

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

    # def handle_error(self, agent_name: str, method: str, error: Exception) -> str:
    #     """
    #     Logs error and returns a formatted message.

    #     Args:
    #         agent_name (str): The name of the agent class
    #         method (str): The method where the error occurred
    #         error (Exception): The exception instance

    #     Returns:
    #         str: User-facing or system-friendly error message
    #     """
    #     error_trace = traceback.format_exc()
    #     self.logger.error(
    #         f"[{agent_name}.{method}] Error: {str(error)}\nTraceback:\n{error_trace}"
    #     )
    #     return f"[{agent_name}] encountered an issue in `{method}`: {str(error)}"
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

    # def safe_execute(self, func, *args, **kwargs):
    #     try:
    #         return func(*args, **kwargs)
    #     except Exception as e:
    #         # Optional: log exception with traceback and context
    #         print(f"[ErrorHandler] Exception: {str(e)}")
    #         raise HTTPException(status_code=500, detail="Internal Server Error: " + str(e))
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
            
    # async def safe_execute_async(self, func, *args, **kwargs):
    #     try:
    #         return await func(*args, **kwargs)
    #     except Exception as e:
    #         self.handle_error(e)
    #         raise HTTPException(status_code=500, detail=str(e))


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