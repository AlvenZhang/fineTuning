def main(args: dict):
    # 获取输入参数
    extra_info_raw = args.get("extraInfo", "")
    consultation_count_raw = args.get("consultationCount")
    sys_histories_raw = args.get("sysHistories", "[]")
    mock_session_history_raw = args.get("mockSessionHistory", "")
    inner_ext_info_raw = args.get("innerExtInfo", "")

    import json
    from datetime import datetime

    def safe_json_loads(maybe_json):
        if isinstance(maybe_json, (dict, list)):
            return maybe_json
        if not isinstance(maybe_json, str) or not maybe_json.strip():
            return None
        text = maybe_json.strip()
        # Try multiple repair strategies
        candidates = [
            text,
            text.replace('\\"', '"'),
            text.strip('"'),
            text.replace('\\"', '"').strip('"'),
        ]
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except:
                pass
        # Try fixing nested JSON in message fields
        try:
            import re
            # Find "message": "{...}" patterns and escape inner quotes
            def fix_message_value(match):
                prefix = match.group(1)  # "message": "
                json_str = match.group(2)  # {...}
                suffix = match.group(3)  # "
                # Escape all quotes inside the JSON string
                escaped = json_str.replace('\\', '\\\\').replace('"', '\\"')
                return prefix + escaped + suffix

            # Match "message": "{...}" where inner JSON may have unescaped quotes
            # Use non-greedy match and handle nested braces
            pattern = r'("message"\s*:\s*")((?:\{(?:[^{}"]|"[^"]*")*\}))(")'
            fixed = re.sub(pattern, fix_message_value, text)
            return json.loads(fixed)
        except:
            pass
        return None

    def normalize_role(type_value):
        try:
            t = int(type_value)
        except:
            t = 1
        return "user" if t == 0 else "assistant"

    def truthy(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes", "y")
        return False

    def extract_content(item):
        if not isinstance(item, dict):
            return ""
        msg = item.get("answer", "")
        if not msg:
            msg = item.get("message", "")

        # For assistant messages (type==1), check if message is JSON
        if item.get("type") == 1 and isinstance(msg, str):
            nested = safe_json_loads(msg)
            if isinstance(nested, dict):
                # Return the full JSON object as minified single-line string
                try:
                    return json.dumps(nested, ensure_ascii=False, separators=(',', ':'))
                except:
                    pass
            # Not JSON, return raw message
            return msg.strip()

        # For user messages or fallback, extract text only
        nested = safe_json_loads(msg) if isinstance(msg, str) else None
        if isinstance(nested, dict):
            msg = nested.get("answer") or nested.get("message") or nested.get("text") or ""
        elif isinstance(msg, dict):
            msg = msg.get("answer") or msg.get("message") or msg.get("text") or ""
        if not isinstance(msg, str):
            try:
                msg = str(msg)
            except:
                msg = ""
        msg = msg.replace(""", '"').replace(""", '"').strip()
        return msg

    def normalize_llm_messages_from_mock(mock_raw):
        parsed = safe_json_loads(mock_raw)
        if not isinstance(parsed, list):
            return None
        out = []
        for it in parsed:
            if not isinstance(it, dict):
                continue
            role = it.get("role")
            content = it.get("content") or it.get("message") or it.get("text")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                msg = {"role": role, "content": content.strip()}
                out.append(msg)
        return out or None

    def normalize_llm_messages_from_sys(sys_raw):
        histories = safe_json_loads(sys_raw)
        if not isinstance(histories, list):
            return []
        out = []
        for item in histories:
            if not isinstance(item, dict):
                continue
            if not truthy(item.get("setMessage", False)):
                continue
            role = normalize_role(item.get("type", 1))
            content = extract_content(item)
            if content:
                msg = {"role": role, "content": content}
                out.append(msg)
        return out

    try:
        data = json.loads(extra_info_raw) if extra_info_raw else {}
    except json.JSONDecodeError:
        data = {}

    # 先试图解析innerExtInfo中的内部变量，如果其中的consultationCount有值且是数字，使用改字段，否则走原逻辑
    try:
        if not inner_ext_info_raw:
            inner_ext_info = {}
        else:
            info = json.loads(inner_ext_info_raw)
            inner_ext_info = info if isinstance(info, dict) else {}
    except json.JSONDecodeError:
        inner_ext_info = {}
    inner_ext_consultation_count = inner_ext_info.get("consultationCount")
    if inner_ext_consultation_count is not None and str(inner_ext_consultation_count).isdigit():
        consultation_count = inner_ext_consultation_count
    else:
        if consultation_count_raw is None or consultation_count_raw == "":
            consultation_count = "1"
        else:
            try:
                consultation_count = str(int(consultation_count_raw) + 1)
            except ValueError:
                consultation_count = "1"

    session_history = normalize_llm_messages_from_mock(mock_session_history_raw) or normalize_llm_messages_from_sys(
        sys_histories_raw)

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Generate sessionHistory_str
    session_history_str = ""
    if session_history:
        for msg in session_history:
            role = "Human" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")
            session_history_str += f"{role}: {content}\n"

    # Parse and validate orderId
    def validate_order_id(value):
        """Validate orderId: must be pure numeric string"""
        if value and isinstance(value, (str, int)):
            value_str = str(value).strip()
            if value_str and value_str.isdigit():
                return value_str
        return None

    validated_order_id = validate_order_id(args.get("globalOrderId", ""))
    if not validated_order_id:
        validated_order_id = validate_order_id(data.get("orderId", ""))

    output = {
        "visitId": data.get("visitId", ""),
        "taskName": data.get("taskName", ""),
        "currentTime": current_time,
        "consultationCount": str(consultation_count),
        "sessionHistory": session_history,
        "sessionHistory_str": session_history_str.strip(),
        "queryOrderId": validated_order_id,
        "sessionId": data.get("sessionId", "")
    }
    return output


if __name__ == "__main__":
    i = {
  "mockSessionHistory": "",
  "globalOrderId": "",
  "consultationCount": "0",
  "sysHistories": "null",
  "innerExtInfo": "",
  "extraInfo": "{  \"queryUid\": \"1985891436121960479_1762308412894\",  \"taskName\": \"咨询行李额-Agent\",  \"taskNodeName\": \"大模型组件\",  \"sessionId\": \"1985891436121960479\",  \"visitId\": \"access-583c4f0b-69e9-4757-b8b4-c8f8ef309479\",  \"taskProcessInstanceId\": \"4d845cef-08a7-441f-84f6-6b4362968732\",  \"buId\": \"14\",  \"subBuId\": \"41\",  \"orderId\": \"78725102914025529360839\",  \"userId\": \"512119113\",  \"userType\": \"40\",  \"taskPath\": \"online\",  \"channel\": \"online\"}"
}
    print(main(i))
    # print(json.loads(i.get("extraInfo")))