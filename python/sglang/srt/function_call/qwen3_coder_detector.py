import ast
import json
import logging
import re
from typing import Any, List, Optional, Tuple

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)

logger = logging.getLogger(__name__)

# Parameter types whose raw value is streamed incrementally as a JSON string:
# chunks are JSON-escaped and wrapped in quotes (`"key": "..."`).
_STRING_PARAM_TYPES = {"string", "str", "text", "varchar", "char", "enum"}

# Parameter types whose raw value is itself valid JSON (array / object). These
# are streamed as raw passthrough chunks (`"key": <raw>`), without quote
# wrapping or escaping, so large structured values (e.g. an `array` of objects)
# can be emitted incrementally instead of being buffered until the terminator.
_JSON_PARAM_TYPES = {"object", "array", "arr"}

# Streaming modes for a parameter value.
_STREAM_MODE_STRING = "string"  # quote-wrapped, JSON-escaped chunks
_STREAM_MODE_JSON = "json"  # raw JSON passthrough chunks


class Qwen3CoderDetector(BaseFormatDetector):
    def __init__(self):
        super().__init__()

        # Sentinel tokens
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_prefix: str = "<function="
        self.function_end_token: str = "</function>"
        self.parameter_prefix: str = "<parameter="
        self.parameter_end_token: str = "</parameter>"

        # Regex for non-streaming fallback
        self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        self.tool_call_function_regex = re.compile(
            r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL
        )
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
            re.DOTALL,
        )

        # Streaming State
        # Base class already initializes _buffer, we just use it directly
        # No need to check with hasattr - we control the lifecycle through inheritance

        # Index pointing to the next character to be processed in buffer
        self.parsed_pos: int = 0
        # Parameter count inside the current tool being processed, used to determine whether to add comma
        self.current_tool_param_count: int = 0
        # Flag indicating whether current tool has already sent '{'
        self.json_started: bool = False

        # [FIX] New state flag: mark whether inside tool_call structure block
        self.is_inside_tool_call: bool = False

        # Initialize attributes that were missing in the original PR
        self.current_func_name: Optional[str] = None

        # Incremental streaming state for a parameter value. Once we cross the
        # opening `<parameter=name>` we emit the `"key": ...` opener and then
        # stream chunks of the value as they arrive:
        #   - string mode: emit `"key": "`, escaped chunks, closing `"`.
        #   - json mode:   emit `"key": `, raw JSON chunks (no quotes/escaping).
        self._streaming_param_name: Optional[str] = None
        self._streaming_param_mode: Optional[str] = None
        self._streaming_prefix_sent: bool = False
        self._streaming_leading_stripped: bool = False

        # Tag prefixes used to guard a streamed chunk against cutting into
        # the middle of an (unfinished) structural tag.
        self._known_tag_prefixes: Tuple[str, ...] = (
            self.tool_call_start_token,
            self.tool_call_end_token,
            self.tool_call_prefix,
            self.function_end_token,
            self.parameter_prefix,
            self.parameter_end_token,
        )

    def has_tool_call(self, text: str) -> bool:
        return self.tool_call_start_token in text

    def _get_arguments_config(
        self, func_name: str, tools: Optional[list[Tool]]
    ) -> dict:
        """Extract argument configuration for a function."""
        if tools is None:
            return {}
        for config in tools:
            try:
                config_type = config.type
                config_function = config.function
                config_function_name = config_function.name
            except AttributeError:
                continue

            if config_type == "function" and config_function_name == func_name:
                try:
                    params = config_function.parameters
                except AttributeError:
                    return {}

                if isinstance(params, dict) and "properties" in params:
                    return params["properties"]
                elif isinstance(params, dict):
                    return params
                else:
                    return {}
        logger.warning(f"Tool '{func_name}' is not defined in the tools list.")
        return {}

    def _convert_param_value(
        self, param_value: str, param_name: str, param_config: dict, func_name: str
    ) -> Any:
        """Convert parameter value based on its type in the schema."""
        # Handle null value for any type
        if param_value.lower() == "null":
            return None

        if param_name not in param_config:
            if param_config != {}:
                logger.warning(
                    f"Parsed parameter '{param_name}' is not defined in the tool "
                    f"parameters for tool '{func_name}', directly returning the string value."
                )
            return param_value

        if (
            isinstance(param_config[param_name], dict)
            and "type" in param_config[param_name]
        ):
            param_type = str(param_config[param_name]["type"]).strip().lower()
        else:
            param_type = "string"
        if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
            return param_value
        elif (
            param_type.startswith("int")
            or param_type.startswith("uint")
            or param_type.startswith("long")
            or param_type.startswith("short")
            or param_type.startswith("unsigned")
        ):
            try:
                param_value = int(param_value)
            except Exception:
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not an integer in tool "
                    f"'{func_name}', degenerating to string."
                )
            return param_value
        elif param_type.startswith("num") or param_type.startswith("float"):
            try:
                maybe_convert = (
                    False if "." in param_value or "e" in param_value.lower() else True
                )
                param_value: float = float(param_value)
                if maybe_convert and param_value.is_integer():
                    param_value = int(param_value)
            except Exception:
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not a float in tool "
                    f"'{func_name}', degenerating to string."
                )
            return param_value
        elif param_type in ["boolean", "bool", "binary"]:
            param_value = param_value.lower()
            if param_value not in ["true", "false"]:
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not a boolean (`true` of `false`) in tool '{func_name}', degenerating to false."
                )
            return param_value == "true"
        else:
            if (
                param_type in ["object", "array", "arr"]
                or param_type.startswith("dict")
                or param_type.startswith("list")
            ):
                try:
                    param_value = json.loads(param_value)
                    return param_value
                except Exception:
                    logger.warning(
                        f"Parsed value '{param_value}' of parameter '{param_name}' cannot be parsed with json.loads in tool "
                        f"'{func_name}', will try other methods to parse it."
                    )
            try:
                param_value = ast.literal_eval(param_value)  # safer
            except Exception:
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' cannot be converted via Python `ast.literal_eval()` in tool '{func_name}', degenerating to string."
                )
            return param_value

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """One-shot parsing for non-streaming scenarios."""
        if self.tool_call_start_token not in text:
            return StreamingParseResult(normal_text=text)

        calls = []
        try:
            # Simple cleanup of the text to find tool calls
            # Note: This is a simplified regex approach consistent with vLLM
            raw_tool_calls = self.tool_call_regex.findall(text)
            if not raw_tool_calls:
                # Fallback: maybe the whole text is inside the tag or tags are stripped
                if self.tool_call_prefix in text:
                    raw_tool_calls = [text]

            tool_idx = 0
            for tool_content in raw_tool_calls:
                # Find function calls
                funcs = self.tool_call_function_regex.findall(tool_content)
                for func_match in funcs:
                    func_body = func_match[0] or func_match[1]
                    if ">" not in func_body:
                        continue

                    name_end = func_body.index(">")
                    func_name = func_body[:name_end]
                    params_str = func_body[name_end + 1 :]

                    param_config = self._get_arguments_config(func_name, tools)
                    parsed_params = {}

                    for p_match in self.tool_call_parameter_regex.findall(params_str):
                        if ">" not in p_match:
                            continue
                        p_idx = p_match.index(">")
                        p_name = p_match[:p_idx]
                        p_val = p_match[p_idx + 1 :]
                        # Remove prefixing and trailing \n
                        if p_val.startswith("\n"):
                            p_val = p_val[1:]
                        if p_val.endswith("\n"):
                            p_val = p_val[:-1]

                        parsed_params[p_name] = self._convert_param_value(
                            p_val, p_name, param_config, func_name
                        )

                    calls.append(
                        ToolCallItem(
                            tool_index=tool_idx,
                            name=func_name,
                            parameters=json.dumps(parsed_params, ensure_ascii=False),
                        )
                    )
                    tool_idx += 1

            # Determine normal text (text before the first tool call)
            start_idx = text.find(self.tool_call_start_token)
            if start_idx == -1:
                start_idx = text.find(self.tool_call_prefix)
            normal_text = text[:start_idx] if start_idx > 0 else ""

            return StreamingParseResult(normal_text=normal_text, calls=calls)

        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            return StreamingParseResult(normal_text=text)

    def _param_stream_mode(
        self, param_name: str, tools: List[Tool]
    ) -> Optional[str]:
        """Resolve how a parameter value should be streamed.

        Returns:
            _STREAM_MODE_STRING - stream as a quote-wrapped, JSON-escaped string.
            _STREAM_MODE_JSON   - stream as raw JSON passthrough (array/object).
            None                - not streamable; buffer the full value and
                                  type-convert it before emission (scalars).
        """
        config = self._get_arguments_config(self.current_func_name, tools)
        if not config or param_name not in config:
            # Unknown parameter defaults to string so text is still streamed.
            return _STREAM_MODE_STRING
        param_schema = config.get(param_name, {})
        if isinstance(param_schema, dict) and "type" in param_schema:
            param_type = str(param_schema["type"]).strip().lower()
        else:
            param_type = "string"
        if param_type in _STRING_PARAM_TYPES:
            return _STREAM_MODE_STRING
        if (
            param_type in _JSON_PARAM_TYPES
            or param_type.startswith("dict")
            or param_type.startswith("list")
        ):
            return _STREAM_MODE_JSON
        return None

    def _find_earliest_param_end(
        self, text: str
    ) -> Optional[Tuple[int, int]]:
        """Earliest parameter-value terminator position in text.

        Returns (position, consumed_length). consumed_length is 0 for the
        "abnormal" terminators (next <parameter= or </function>) so that the
        caller can re-process that tag on the next iteration.
        """
        candidates: List[Tuple[int, int]] = []
        end_param = text.find(self.parameter_end_token)
        if end_param != -1:
            candidates.append((end_param, len(self.parameter_end_token)))
        next_param = text.find(self.parameter_prefix)
        if next_param != -1:
            candidates.append((next_param, 0))
        end_func = text.find(self.function_end_token)
        if end_func != -1:
            candidates.append((end_func, 0))
        if not candidates:
            return None
        return min(candidates, key=lambda x: x[0])

    def _safe_stream_len(self, text: str) -> int:
        """Length of text safe to emit without cutting a partial tag prefix."""
        n = len(text)
        if n == 0:
            return 0
        search_from = 0
        while search_from < n:
            lt_pos = text.find("<", search_from)
            if lt_pos == -1:
                return n
            suffix = text[lt_pos:]
            # If the suffix starting at this '<' could still be completed
            # into a known tag, treat it as ambiguous and back off.
            for tag in self._known_tag_prefixes:
                if tag.startswith(suffix):
                    return lt_pos
            # This '<' cannot start any known tag - it is literal content.
            search_from = lt_pos + 1
        return n

    def _emit_param_prefix(
        self, param_name: str, mode: str, calls: List[ToolCallItem]
    ) -> None:
        """Emit '{' (if needed) plus the '"key": ' opener for a value.

        For string mode the opener also includes the opening quote so value
        chunks can be appended directly; for json mode the raw value follows.
        """
        if not self.json_started:
            calls.append(
                ToolCallItem(tool_index=self.current_tool_id, parameters="{")
            )
            self.json_started = True

        # Opening quote only for string mode; json values are raw passthrough.
        opener = '"' if mode == _STREAM_MODE_STRING else ""
        if self.current_tool_param_count > 0:
            prefix = f", {json.dumps(param_name)}: {opener}"
        else:
            prefix = f"{json.dumps(param_name)}: {opener}"
        calls.append(
            ToolCallItem(tool_index=self.current_tool_id, parameters=prefix)
        )
        self.current_tool_param_count += 1
        self._streaming_prefix_sent = True

    def _encode_value_chunk(self, chunk: str, mode: str) -> str:
        """Encode a raw value chunk for emission according to the stream mode.

        String mode JSON-escapes the chunk (it lives inside `"..."`); json mode
        passes it through verbatim, since the value is already valid JSON.
        """
        if mode == _STREAM_MODE_STRING:
            return json.dumps(chunk, ensure_ascii=False)[1:-1]
        return chunk

    def _reset_param_streaming_state(self) -> None:
        self._streaming_param_name = None
        self._streaming_param_mode = None
        self._streaming_prefix_sent = False
        self._streaming_leading_stripped = False

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Robust cursor-based streaming parser. String-typed parameter values
        are streamed incrementally as their raw chunks arrive; other types
        are buffered until the parameter terminator is seen so the value can
        be properly type-converted before emission.
        """
        self._buffer += new_text

        # Guard against empty buffer
        if not self._buffer:
            return StreamingParseResult()

        calls = []
        normal_text_chunks = []

        while True:
            # Working text slice
            current_slice = self._buffer[self.parsed_pos :]

            # Optimization: If almost empty, wait for more
            if not current_slice:
                break

            # -------------------------------------------------------
            # 0. Active incremental string parameter value streaming
            # -------------------------------------------------------
            if self._streaming_param_name is not None:
                mode = self._streaming_param_mode

                # Strip a single leading newline once, at the very start of
                # the value (before any content has been emitted).
                if not self._streaming_leading_stripped:
                    if current_slice[0] == "\n":
                        self.parsed_pos += 1
                        self._streaming_leading_stripped = True
                        continue
                    self._streaming_leading_stripped = True
                    # Fall through with the same slice.

                end_info = self._find_earliest_param_end(current_slice)

                if end_info is not None:
                    end_pos, end_len = end_info
                    raw_value = current_slice[:end_pos]
                    # Mirror the non-streaming strip of a single trailing \n.
                    if raw_value.endswith("\n"):
                        raw_value = raw_value[:-1]

                    # Whether any value content has already been streamed. If
                    # the prefix is still unsent, the whole value is empty.
                    value_streamed = self._streaming_prefix_sent
                    if not self._streaming_prefix_sent:
                        self._emit_param_prefix(
                            self._streaming_param_name, mode, calls
                        )

                    if raw_value:
                        chunk_out = self._encode_value_chunk(raw_value, mode)
                        if chunk_out:
                            calls.append(
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    parameters=chunk_out,
                                )
                            )
                    elif mode == _STREAM_MODE_JSON and not value_streamed:
                        # Empty json value with nothing streamed yet: the prefix
                        # `"key": ` was just emitted with no quotes, so emit a
                        # valid literal to keep the stream valid JSON. (Guarded
                        # by `not value_streamed` so a terminator slice that only
                        # holds the stripped trailing newline of an already-
                        # streamed value does not append a spurious `null`.)
                        calls.append(
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                parameters="null",
                            )
                        )
                    # Close the string value; json values need no closer.
                    if mode == _STREAM_MODE_STRING:
                        calls.append(
                            ToolCallItem(
                                tool_index=self.current_tool_id, parameters='"'
                            )
                        )

                    self.parsed_pos += end_pos + end_len
                    self._reset_param_streaming_state()
                    continue

                # No terminator yet - emit what we can safely, holding back
                # any suffix that could be part of an incoming tag.
                safe_len = self._safe_stream_len(current_slice)
                if safe_len > 0:
                    chunk = current_slice[:safe_len]
                    # Hold back a trailing newline: if it turns out to be
                    # the value's trailing newline it must be stripped when
                    # the terminator finally arrives.
                    if chunk.endswith("\n"):
                        chunk = chunk[:-1]

                    if chunk:
                        if not self._streaming_prefix_sent:
                            self._emit_param_prefix(
                                self._streaming_param_name, mode, calls
                            )
                        chunk_out = self._encode_value_chunk(chunk, mode)
                        if chunk_out:
                            calls.append(
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    parameters=chunk_out,
                                )
                            )
                        self.parsed_pos += len(chunk)
                break

            # -------------------------------------------------------
            # 1. Priority detection: check if it's the start of Tool Call
            # -------------------------------------------------------
            if current_slice.startswith(self.tool_call_start_token):
                self.parsed_pos += len(self.tool_call_start_token)
                self.is_inside_tool_call = True
                continue

            # -------------------------------------------------------
            # 2. Function Name: <function=name>
            # -------------------------------------------------------
            if current_slice.startswith(self.tool_call_prefix):
                end_angle = current_slice.find(">")
                if end_angle != -1:
                    func_name = current_slice[len(self.tool_call_prefix) : end_angle]

                    self.current_tool_id += 1
                    self.current_tool_name_sent = True
                    self.current_tool_param_count = 0
                    self.json_started = False
                    self.current_func_name = func_name

                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=func_name,
                            parameters="",
                        )
                    )

                    self.parsed_pos += end_angle + 1
                    continue
                else:
                    # Incomplete tag
                    break

            # -------------------------------------------------------
            # 3. Parameter: <parameter=name>value...
            # -------------------------------------------------------
            if current_slice.startswith(self.parameter_prefix):
                name_end = current_slice.find(">")
                if name_end != -1:
                    param_name = current_slice[
                        len(self.parameter_prefix) : name_end
                    ]
                    value_start_idx = name_end + 1
                    rest_of_slice = current_slice[value_start_idx:]

                    stream_mode = self._param_stream_mode(param_name, tools)
                    if stream_mode is not None:
                        # Enter incremental streaming mode (string or json).
                        # The value is emitted chunk-by-chunk in branch 0 above.
                        self.parsed_pos += value_start_idx
                        self._streaming_param_name = param_name
                        self._streaming_param_mode = stream_mode
                        self._streaming_prefix_sent = False
                        self._streaming_leading_stripped = False
                        continue

                    # Non-streamable scalar: buffer the full value so it can be
                    # type converted before emission.
                    end_info = self._find_earliest_param_end(rest_of_slice)
                    if end_info is None:
                        break

                    end_pos, end_len = end_info
                    raw_value = rest_of_slice[:end_pos]

                    # Cleanup value
                    if raw_value.startswith("\n"):
                        raw_value = raw_value[1:]
                    if raw_value.endswith("\n"):
                        raw_value = raw_value[:-1]

                    # JSON Construction
                    if not self.json_started:
                        calls.append(
                            ToolCallItem(
                                tool_index=self.current_tool_id, parameters="{"
                            )
                        )
                        self.json_started = True

                    param_config = self._get_arguments_config(
                        self.current_func_name, tools
                    )
                    converted_val = self._convert_param_value(
                        raw_value, param_name, param_config, self.current_func_name
                    )

                    # Construct JSON fragment: "key": value
                    # Note: We must be careful with json.dumps to ensure valid JSON streaming
                    json_key_val = f"{json.dumps(param_name)}: {json.dumps(converted_val, ensure_ascii=False)}"

                    if self.current_tool_param_count > 0:
                        fragment = f", {json_key_val}"
                    else:
                        fragment = json_key_val

                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id, parameters=fragment
                        )
                    )
                    self.current_tool_param_count += 1

                    # Advance cursor
                    total_len = (name_end + 1) + end_pos + end_len
                    self.parsed_pos += total_len
                    continue

                # Incomplete parameter tag
                break

            # -------------------------------------------------------
            # 4. Function End: </function>
            # -------------------------------------------------------
            if current_slice.startswith(self.function_end_token):
                if not self.json_started:
                    calls.append(
                        ToolCallItem(tool_index=self.current_tool_id, parameters="{")
                    )
                    self.json_started = True

                calls.append(
                    ToolCallItem(tool_index=self.current_tool_id, parameters="}")
                )
                self.parsed_pos += len(self.function_end_token)
                self.current_func_name = None
                continue

            # -------------------------------------------------------
            # 5. Tool Call End: </tool_call>
            # -------------------------------------------------------
            if current_slice.startswith(self.tool_call_end_token):
                self.parsed_pos += len(self.tool_call_end_token)
                self.is_inside_tool_call = False  # [FIX] Exit tool call region
                continue

            # -------------------------------------------------------
            # 6. Handling content / whitespace / normal text
            # -------------------------------------------------------
            # If current position is not the start of a tag (i.e., doesn't start with <), it might be plain text,
            # or a newline between two tags.
            # But we need to be careful not to output truncated tags like "<fun" as text.

            next_open_angle = current_slice.find("<")

            if next_open_angle == -1:
                # This entire segment is plain text
                if not self.is_inside_tool_call:
                    normal_text_chunks.append(current_slice)
                # [FIX] If inside tool call, discard this text (usually \n), don't append
                self.parsed_pos += len(current_slice)
                continue

            elif next_open_angle == 0:
                # Looks like a Tag, but doesn't match any known Tag above

                is_potential_tag = False
                for tag in self._known_tag_prefixes:
                    if tag.startswith(current_slice):
                        is_potential_tag = True
                        break

                if is_potential_tag:
                    break  # Wait for more
                else:
                    # Just a plain '<' symbol
                    if not self.is_inside_tool_call:
                        normal_text_chunks.append("<")
                    self.parsed_pos += 1
                    continue

            else:
                # '<' is in the middle
                text_segment = current_slice[:next_open_angle]
                if not self.is_inside_tool_call:
                    normal_text_chunks.append(text_segment)
                # [FIX] If inside tool call, discard whitespace/text before Tag
                self.parsed_pos += next_open_angle
                continue

        # Memory Cleanup: Slice the buffer
        # Keep unparsed part, discard parsed part
        if self.parsed_pos > 0:
            self._buffer = self._buffer[self.parsed_pos :]
            self.parsed_pos = 0

        normal_text = "".join(normal_text_chunks) if normal_text_chunks else ""
        return StreamingParseResult(calls=calls, normal_text=normal_text)

    def supports_structural_tag(self) -> bool:
        return True

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError

    def get_structural_tag_name(self) -> str:
        return "qwen_3_coder"
