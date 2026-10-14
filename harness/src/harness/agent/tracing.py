"""Optional LangSmith spans over public runtime data, independent of audit events.

No Harness, provider configuration, raw response, or evaluated SessionSnapshot
is serialized. Export failures never retry an operation or change its result.
"""

from functools import wraps
import logging
import os


logger = logging.getLogger(__name__)


def tracing_enabled():
    value = os.environ.get("LANGSMITH_TRACING")
    if value is None:
        value = os.environ.get("LANGCHAIN_TRACING_V2", "false")
    return value.lower() == "true"


def require_tracing():
    if tracing_enabled():
        try:
            import langsmith  # noqa: F401
        except ImportError as exc:
            raise ImportError("LangSmith tracing requires pip install '.[tracing]'") from exc


def public_outcome(result):
    """An explicit allowlist: evaluation and private labels stay in local artifacts."""
    return {
        "status": result.status,
        "terminal_reason": result.terminal_reason,
        "action_count": result.action_count,
        "search_count": result.search_count,
        "model_turn_count": result.model_turn_count,
        "tool_call_count": result.tool_call_count,
        "usage": result.usage.model_dump(mode="json") if result.usage else None,
    }


def _inputs(kind, owner, args):
    if kind == "model":
        messages, definitions = args
        return {
            "messages": [m.as_provider_payload() for m in messages],
            "tools": [d.as_provider_payload() for d in definitions],
        }, {
            "ls_provider": owner.provider.name,
            "ls_model_name": owner.provider.model,
            "ls_model_type": "chat",
        }
    metadata = {
        "session_id": owner.session_id,
        "thread_id": owner.session_id,
        "task_id": owner.task.task.task_id,
        "agent_role": owner.coordinator.role if owner.coordinator else "single",
        "orchestration_backend": owner.owner.config.orchestration_backend,
    }
    inputs = {"budget": owner.budget().model_dump(mode="json")}
    if kind == "run":
        inputs["task"] = owner.task.model_dump(mode="json")
    if kind == "tool":
        inputs["action"] = args[0].model_dump(mode="json")
    return inputs, metadata


def _outputs(kind, owner, result):
    if kind == "model":
        output = {"message": result.as_message().as_provider_payload()}
        if result.usage:
            output["usage_metadata"] = {
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
                "total_tokens": result.usage.total_tokens,
            }
        return output
    if kind == "tool":
        return result.model_dump(mode="json")
    output = {"budget": owner.budget().model_dump(mode="json")}
    if result is not None:
        output.update(public_outcome(result))
    return output


def traced(kind):
    """Instrument only known public boundaries; never serialize self or raw results."""
    def decorate(function):
        @wraps(function)
        def wrapped(owner, *args, **kwargs):
            if not tracing_enabled():
                return function(owner, *args, **kwargs)
            manager = span = None
            try:
                from langsmith import trace

                inputs, metadata = _inputs(kind, owner, args)
                name = f"shopping.{kind}"
                if kind == "turn":
                    name = f"shopping.{metadata['agent_role']}"
                elif kind == "tool":
                    name = f"shopping.tool.{args[0].name}"
                manager = trace(
                    name,
                    run_type={"model": "llm", "tool": "tool"}.get(kind, "chain"),
                    inputs=inputs,
                    metadata=metadata,
                    tags=["shopping-simulator"],
                )
                span = manager.__enter__()
            except Exception:
                manager = None
                logger.warning("LangSmith span could not start; continuing without this span")
            try:
                result = function(owner, *args, **kwargs)
            except BaseException as exc:
                if manager is not None:
                    try:
                        manager.__exit__(type(exc), exc, exc.__traceback__)
                    except Exception:
                        logger.warning("LangSmith error span could not close")
                raise
            else:
                if manager is not None:
                    try:
                        if span is not None:
                            span.end(outputs=_outputs(kind, owner, result))
                    except Exception:
                        logger.warning("LangSmith output could not be recorded")
                    finally:
                        try:
                            manager.__exit__(None, None, None)
                        except Exception:
                            logger.warning("LangSmith span could not close")
                return result
        return wrapped
    return decorate
