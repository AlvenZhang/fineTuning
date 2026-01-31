import json
from datetime import datetime
def main(args: dict):
    # 获取输入参数
    mock_session_history_raw = "[{\"role\": \"assistant\",\"message\": \"您好，目前是否下单了呢\"},{\"role\": \"user\",\"message\": \"还没用\"},{\"role\": \"assistant\",\"message\": \"具体想咨询什么问题呢\"},{\"role\": \"user\",\"message\": \"我的行李箱是45*30*20的\"}]"
    # print(safe_json_loads(json1))
    extra_info_raw = args.get("extraInfo", "")
    consultation_count_raw = args.get("consultationCount")
    sys_histories_raw = args.get("sysHistories", "[]")
    # mock_session_history_raw = args.get("mockSessionHistory", "")

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

    if consultation_count_raw is None or consultation_count_raw == "":
        consultation_count = "1"
    else:
        try:
            consultation_count = str(int(str(consultation_count_raw)) + 1)
        except:
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

    output = {
        "visitId": data.get("visitId", ""),
        "taskName": data.get("taskName", ""),
        "currentTime": current_time,
        "consultationCount": str(consultation_count),
        "sessionHistory": session_history,
        "sessionHistory_str": session_history_str.strip()
    }
    return output

if __name__ == "__main__":
    json1 = '[{"role": "System","message": "您好，目前是否下单了呢"},{"role": "User","message": "还没用"},{"role": "System","message": "具体想咨询什么问题呢"},{"role": "User","message": "我的行李箱是45*30*20的"}]'
    print(main({}))
    # j = json.loads(json1)
    # print(j)