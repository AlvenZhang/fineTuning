def main(args: dict):
    import json
    cardtype = args.get("cardtype", "")
    cardtypeUsageCount = args.get("cardtypeUsageCount", "")

    def safe_json_loads(maybe_json):
        if isinstance(maybe_json, (dict, list)):
            return maybe_json
        if not isinstance(maybe_json, str) or not maybe_json.strip():
            return {}
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
                return {}
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
            return {}


    cardtypeUsageCount = safe_json_loads(cardtypeUsageCount)
    try:
        if cardtype in cardtypeUsageCount:
            # cardtype只允许使用一次
            cardtypeUsageCount[cardtype] += 1
            if cardtypeUsageCount[cardtype] > 1:
                cardtype = ""
        else:
            cardtypeUsageCount[cardtype] = 1
        return {
            "cardtypeUsageCount": json.dumps(cardtypeUsageCount, ensure_ascii=False),
            "cardtype": cardtype
        }
    except:
        return {
            "cardtypeUsageCount": "",
            "cardtype": ""
        }

if __name__ == "__main__":
    d = {
        "cardtype": "展示查询行李规定入口",
        "cardtypeUsageCount": ''
    }
    res1 = main(d)
    print(res1)
    # d2 = {
    #     "cardtype":"行李额2",
    #     "cardtypeUsageCount": '{"行李额": 3}'
    # }
    # print(main(d2))
