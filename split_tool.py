def main(args: dict):
    answer = args.get("answer", "").strip()
    cardType = args.get("cardType", "").strip()

    ret = {
        "needSplit": "false",
        "splitAnswer_1": "",
        "splitAnswer_2": "",
        "splitAnswer_end": "",
        "answer": answer
    }

    if not answer:
        return ret

    # 定义标点符号集合
    punctuations = "。！？；，：,.!?;:"

    # 分隔符处理逻辑
    def custom_split_replace(text):
        parts = text.split("<split>")
        result = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if part[-1] in punctuations:
                result.append(part)
            else:
                result.append(part + "。")
        return "".join(result)

    # cardType 有值时，不分割，仅处理分隔符
    if cardType:
        ret["answer"] = custom_split_replace(answer)
        return ret

    # 不含分隔符，不分割，仅处理分隔符
    if "<split>" not in answer:
        ret["answer"] = custom_split_replace(answer)
        return ret

    # 分割并去除空白
    parts = [p.strip() for p in answer.split("<split>") if p.strip()]
    if not parts or len(parts) > 3:
        ret["answer"] = custom_split_replace(answer)
        return ret

    ret["needSplit"] = "true"
    if len(parts) == 1:
        ret["splitAnswer_end"] = parts[0]
    elif len(parts) == 2:
        ret["splitAnswer_1"] = parts[0]
        ret["splitAnswer_end"] = parts[1]
    elif len(parts) == 3:
        ret["splitAnswer_1"] = parts[0]
        ret["splitAnswer_2"] = parts[1]
        ret["splitAnswer_end"] = parts[2]

    ret["answer"] = custom_split_replace(answer)
    return ret


if __name__ == "__main__":
    input = {
      "cardType": "展示查询行李规定入口",
      "answer": "亲，我帮您查了一下，11月19日 成都-大理这一段航班的信息。每位成人可免费携带1件手提行李，限重7KG，体积不超过20×30×40cm（随身带上飞机的行李）<split>托运行李方面，您购买的舱位暂无免费托运行李额。如需托运，建议提前联系航司咨询付费托运事宜。"
    }
    print(main(input))
