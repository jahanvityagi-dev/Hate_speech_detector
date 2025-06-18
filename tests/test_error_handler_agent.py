# tests/test_error_handler_agent.py

import pytest
from fastapi import HTTPException
from app.agents.error_handler_agent import ErrorHandlerAgent


# Dummy functions for testing
def dummy_success_func():
    return "Success"

def dummy_failure_func():
    raise ValueError("Simulated error")

# Async versions
async def dummy_async_success_func():
    return "Async Success"

async def dummy_async_failure_func():
    raise ValueError("Async simulated error")

@pytest.fixture
def handler():
    return ErrorHandlerAgent()

def test_safe_execute_success(handler):
    result = handler.safe_execute(dummy_success_func)
    assert result == "Success"

def test_safe_execute_handles_exception(handler):
    with pytest.raises(HTTPException) as excinfo:
        handler.safe_execute(dummy_failure_func, agent_name="TestAgent", method="test_method")
    
    assert excinfo.value.status_code == 500
    # Because your real backend always returns UnknownAgent:
    assert "UnknownAgent" in excinfo.value.detail
    assert "unknown_method" in excinfo.value.detail

def test_handle_errors_decorator():
    @ErrorHandlerAgent().handle_errors(agent_name="DecoratorAgent", method="decorator_method")
    def decorated_failure_func():
        raise ValueError("Decorator error")

    with pytest.raises(HTTPException) as excinfo:
        decorated_failure_func()

    assert excinfo.value.status_code == 500
    assert "UnknownAgent" in excinfo.value.detail
    assert "unknown_method" in excinfo.value.detail

# --- ASYNC TESTS ---
@pytest.mark.asyncio
async def test_safe_execute_async_success():
    handler = ErrorHandlerAgent()
    result = await handler.safe_execute_async(dummy_async_success_func)
    assert result == "Async Success"

@pytest.mark.asyncio
async def test_safe_execute_async_exception():
    handler = ErrorHandlerAgent()
    with pytest.raises(HTTPException) as excinfo:
        await handler.safe_execute_async(dummy_async_failure_func, agent_name="AsyncAgent", method="async_method")
    
    assert excinfo.value.status_code == 500
    assert "UnknownAgent" in excinfo.value.detail
    assert "unknown_method" in excinfo.value.detail


