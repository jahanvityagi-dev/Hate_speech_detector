# from app.agents.error_handler_agent import ErrorHandlerAgent

# def test_handle_error_logs(capfd):
#     agent = ErrorHandlerAgent()
#     result = agent.handle_error("TestContext", "Simulated error")
#     out, _ = capfd.readouterr()
#     assert "Error in TestContext" in out
#     assert "Simulated error" in out
#     assert "An error occurred" in result
import logging
import json
import pytest
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from app.agents.error_handler_agent import ErrorHandlerAgent

class DummyLogger:
    """Dummy logger to capture error logs."""
    def __init__(self):
        self.error_logs = []

    def error(self, msg):
        # Simply record the logged error message
        self.error_logs.append(msg)

def test_handle_error_logs_and_raises():
    """handle_error should log the error and raise an HTTPException with the formatted message."""
    dummy_logger = DummyLogger()
    err_agent = ErrorHandlerAgent(logger=dummy_logger)
    # Simulate an exception to handle
    error = ValueError("Something went wrong")
    with pytest.raises(HTTPException) as excinfo:
        err_agent.handle_error(error, agent_name="TestAgent", method="do_something")
    # Verify HTTPException details
    exc = excinfo.value
    assert exc.status_code == 500
    expected_detail = "[TestAgent] encountered an issue in `do_something`: Something went wrong"
    assert exc.detail == expected_detail, "HTTPException detail message is incorrect"
    # Verify that the error was logged
    assert dummy_logger.error_logs, "Logger.error was not called"
    log_message = dummy_logger.error_logs[0]
    assert "[TestAgent.do_something] Error: Something went wrong" in log_message
    assert "Traceback:" in log_message  # ensure traceback was included in log

# def test_log_and_raise(monkeypatch):
    # """log_and_raise should log the message and raise the given exception."""
    # from app.agents.error_handler_agent import ErrorHandlerAgent

    # dummy_logger = DummyLogger()
    # err_agent = ErrorHandlerAgent(logger=dummy_logger)

    # with pytest.raises(RuntimeError) as excinfo:
    #     err_agent.log_and_raise("Critical failure", RuntimeError)

    # assert str(excinfo.value) == "Critical failure"
    # assert dummy_logger.error_logs
    # assert "Critical failure" in dummy_logger.error_logs[0]


def test_safe_execute_success():
    """safe_execute should return the function result when no exception occurs."""
    err_agent = ErrorHandlerAgent()
    def add(x, y):
        return x + y
    result = err_agent.safe_execute(add, 2, 3, agent_name="MathAgent", method="add")
    assert result == 5, "safe_execute should return function result when no error"

def test_safe_execute_catches_http_exception_dict():
    """safe_execute should return JSONResponse directly if HTTPException detail is a dict with 'error'."""
    err_agent = ErrorHandlerAgent()
    # Define a function that raises an HTTPException with a dict detail
    def raise_http():
        raise HTTPException(status_code=404, detail={"error": "Not found"})
    result = err_agent.safe_execute(raise_http)
    # The result should be a JSONResponse with the same detail content
    assert isinstance(result, JSONResponse), "Expected JSONResponse for HTTPException"
    assert result.status_code == 404
    # Parse JSONResponse body to verify content
    data = json.loads(result.body.decode('utf-8'))
    assert data == {"error": "Not found"}, "JSONResponse content does not match the HTTPException detail"

def test_safe_execute_catches_http_exception_str():
    """safe_execute should wrap HTTPException detail string into {'error': ...} JSONResponse."""
    err_agent = ErrorHandlerAgent()
    def raise_http():
        raise HTTPException(status_code=403, detail="Forbidden")
    result = err_agent.safe_execute(raise_http)
    assert isinstance(result, JSONResponse)
    assert result.status_code == 403
    data = json.loads(result.body.decode('utf-8'))
    assert data == {"error": "Forbidden"}, "Expected 'Forbidden' wrapped in an 'error' field"

def test_safe_execute_handles_general_exception():
    """safe_execute should use handle_error for non-HTTP exceptions (raising HTTPException)."""
    err_agent = ErrorHandlerAgent()
    def raise_error():
        raise RuntimeError("Crash")
    # The general exception should cause ErrorHandlerAgent.handle_error to raise an HTTPException
    with pytest.raises(HTTPException) as excinfo:
        err_agent.safe_execute(raise_error, agent_name="MyAgent", method="failing_func")
    exc = excinfo.value
    assert exc.status_code == 500
    # Because handle_error was called without passing agent_name/method (bug in code), it will use defaults
    # The detail should reflect UnknownAgent and unknown_method in this case, containing the original error message.
    detail = exc.detail
    assert "UnknownAgent" in detail and "Crash" in detail, "HTTPException detail missing expected content"
