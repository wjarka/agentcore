import logging
from typing import Any, cast, override

from ..base import (
    BaseProvider,
    BaseSpanTypes,
    ProviderBehavior,
    SpanKind,
)


class IndentedFormatter(logging.Formatter):
    """Custom formatter that maintains indentation across multilines."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    @override
    def format(self, record: logging.LogRecord) -> str:
        # Get the base formatted message
        message = super().format(record)

        # Get indentation from the record (set by our sink)
        indent = getattr(record, "span_indent", "")

        if indent and "\n" in message:
            # Split message into lines and indent each one
            lines = message.split("\n")
            indented_lines = [lines[0]]  # First line already has indent from our sink
            for line in lines[1:]:
                indented_lines.append(indent + line)
            message = "\n".join(indented_lines)

        return message


class IndentedLoggerBehavior(ProviderBehavior):
    """Logger sink that respects existing logger configuration."""

    def __init__(
        self,
        logger: logging.Logger | None = None,
        max_text_length: int = 200,
        use_custom_formatting: bool = False,
        logger_name: str = "telemetry",
    ):
        self.max_text_length: int = max_text_length
        self.use_custom_formatting: bool = use_custom_formatting

        if logger is None:
            # Create our own logger - won't interfere with anything
            self.logger: logging.Logger = logging.getLogger(logger_name)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                if use_custom_formatting:
                    formatter = IndentedFormatter(
                        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    )
                    handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
        else:
            # Use provided logger but DON'T modify its configuration
            self.logger = logger

            # Check if it already has the custom formatter
            has_custom_formatter = any(
                isinstance(h.formatter, IndentedFormatter) for h in self.logger.handlers
            )

            if use_custom_formatting and not has_custom_formatter:
                # Only add custom formatting if explicitly requested AND not already present
                self._add_custom_handler()

    def _add_custom_handler(self):
        """Add a custom handler without modifying existing ones."""
        # Create a separate handler for telemetry logs
        telemetry_handler = logging.StreamHandler()
        formatter = IndentedFormatter(
            "%(asctime)s - TELEMETRY - %(levelname)s - %(message)s"
        )
        telemetry_handler.setFormatter(formatter)

        # Add filter to only handle telemetry-related logs
        telemetry_handler.addFilter(
            lambda record: getattr(record, "is_telemetry", False)
        )
        self.logger.addHandler(telemetry_handler)

    def _log_with_context(
        self, level: int, message: str, stack: list[Any], **extra: Any
    ):
        """Log with telemetry context information."""
        indent = "  " * (len(stack) - 1)

        # Prepare context data
        context_extra = {
            "span_depth": len(stack),
            "span_path": " -> ".join([s.name for s in stack]),
            "is_telemetry": True,
            **extra,
        }

        if self.use_custom_formatting:
            # Use our custom formatting with indentation
            context_extra["span_indent"] = indent
            formatted_message = f"{indent}{message}"
        else:
            # Use structured logging - add context as extra fields
            formatted_message = message
            # Add indentation as a prefix for readability
            formatted_message = f"{'  ' * (len(stack) - 1)}{message}"

        self.logger.log(level, formatted_message, extra=context_extra)

    def _get_indent(self, stack: list[Any]) -> str:
        """Get indentation string based on stack depth."""
        return "  " * (len(stack) - 1)

    def _format_value(self, value: Any, max_length: int | None = None) -> str:
        """Smart formatting for potentially long values."""
        if max_length is None:
            max_length = self.max_text_length

        if value is None:
            return "None"

        # Convert to string
        str_value = str(value)

        # If it's short enough, return as-is
        if len(str_value) <= max_length:
            return str_value

        # For longer values, provide smart truncation
        if isinstance(value, dict):
            value = cast(dict[Any, Any], value)
            return self._format_dict(value, max_length)
        elif isinstance(value, (list, tuple)):
            value = cast(list[Any] | tuple[Any], value)
            return self._format_sequence(value, max_length)
        else:
            return self._format_long_text(str_value, max_length)

    def _format_dict(self, d: dict[Any, Any], max_length: int) -> str:
        """Format dictionary with smart truncation."""
        if not d:
            return "{}"

        # Try to show structure
        keys = list(d.keys())
        if len(keys) <= 3:
            # Small dict, try to show it all
            short_repr = str(d)
            if len(short_repr) <= max_length:
                return short_repr

        # Large dict, show summary
        return f"{{...}} ({len(keys)} keys: {', '.join(str(k) for k in keys[:3])}{'...' if len(keys) > 3 else ''})"

    def _format_sequence(self, seq: list[Any] | tuple[Any], max_length: int) -> str:
        """Format list/tuple with smart truncation."""
        bracket = "[]" if isinstance(seq, list) else "()"
        if not seq:
            return bracket

        # Try to show first few items
        items_str = ", ".join(
            str(item)[:50] for item in seq[:3]
        )  # Max 50 chars per item
        if len(seq) <= 3 and len(items_str) <= max_length:
            return f"{bracket[0]}{items_str}{bracket[1]}"

        return f"{bracket[0]}...{bracket[1]} ({len(seq)} items)"

    def _format_long_text(self, text: str, max_length: int) -> str:
        """Format long text with smart truncation."""
        lines = text.split("\n")

        if len(lines) > 1:
            # Multiline text
            first_line = lines[0][: max_length // 2] if lines[0] else ""
            return f'"{first_line}..." ({len(lines)} lines, {len(text)} chars)'
        else:
            # Single line, truncate in middle to preserve start and end
            if len(text) <= max_length:
                return f'"{text}"'

            half = (max_length - 5) // 2  # Account for quotes and "..."
            return f'"{text[:half]}...{text[-half:]}" ({len(text)} chars)'

    @override
    def on_enter(self, kind: SpanKind, span: BaseSpanTypes, stack: list[Any]) -> None:
        span_name = span.name

        self._log_with_context(
            logging.INFO,
            f"📍 Starting: {span_name}",
            stack,
            span_name=span_name,
            span_action="enter",
        )

        # Log input if present
        if span.input is not None:
            formatted_input = self._format_value(span.input)
            self._log_with_context(
                logging.INFO,
                f"  📥 Input: {formatted_input}",
                stack,
                span_name=span_name,
                span_input=span.input,
            )

    @override
    def on_exit(
        self,
        kind: SpanKind,
        span: BaseSpanTypes,
        stack: list[Any],
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> bool | None:
        span_name = span.name

        if exc_val:
            self._log_with_context(
                logging.ERROR,
                f"❌ Failed: {span_name} - {exc_val}",
                stack,
                span_name=span_name,
                span_action="exit",
                span_error=str(exc_val),
                span_success=False,
            )
        else:
            self._log_with_context(
                logging.INFO,
                f"✅ Completed: {span_name}",
                stack,
                span_name=span_name,
                span_action="exit",
                span_success=True,
            )

            # Log output if present
            if span.output is not None:
                formatted_output = self._format_value(span.output)
                self._log_with_context(
                    logging.INFO,
                    f"  📤 Output: {formatted_output}",
                    stack,
                    span_name=span_name,
                    span_output=span.output,
                )

        return None


def get_instance(
    logger: logging.Logger | None = None,
    max_text_length: int = 200,
    use_custom_formatting: bool = False,
    logger_name: str = "telemetry",
):
    return BaseProvider(
        provider_behavior=IndentedLoggerBehavior(
            logger=logger,
            max_text_length=max_text_length,
            use_custom_formatting=use_custom_formatting,
            logger_name=logger_name,
        )
    )
