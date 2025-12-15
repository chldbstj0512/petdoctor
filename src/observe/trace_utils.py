from typing import Callable, Dict, Any


def traced_node(name: str, fn: Callable):
    """
    Wrap a LangGraph node with Langfuse span.
    """

    def wrapper(state: Dict[str, Any]):
        trace = state.get("_trace")
        if trace is None:
            return fn(state)

        span = trace.span(name=name)
        try:
            result = fn(state)

            # span metadata (optional)
            span.update(
                metadata={
                    "input_keys": list(state.keys()),
                    "output_keys": list(result.keys()),
                }
            )
            return result

        except Exception as e:
            span.update(
                status="error",
                error=str(e),
            )
            raise

        finally:
            span.end()

    return wrapper
