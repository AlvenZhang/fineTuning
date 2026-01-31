# -*- coding: utf-8 -*-

import json
import datetime


def main(args: dict):
    """
    组装上下文信息：解析订单详情并结合知识库输出为统一文本块

    Args:
        args: 包含以下键的字典:
            - orderId: 订单ID，用于验证订单是否存在
            - orderDetail: 订单详情 JSON 字符串
            - qaList: 知识库内容字符串
            - luggageBizKnowledge: 额外信息收集
            - isWorking: 客服是否在线
            - workTimeBegin: 客服在线开始时间
            - workTimeEnd: 客服在线结束时间

    Returns:
        包含以下键的字典:
            - context: 完整的上下文文本
            - confirm: 订单确认话术
            - needConfirm: 是否需要确认订单
    """
    sections = []
    valid_order_id = args.get("orderId", None)
    order_detail_str = args.get("orderDetail", None)
    confirm_text = None
    # 获取当前日期时间
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    sections.append(f"#### 当前时间：{current_time}")

    # 检查订单ID是否有效
    if not valid_order_id or valid_order_id == "":
        # 场景 A：完全没有订单
        sections.append(
            "#### 当前用户订单信息:\n**订单状态**：没有订单\n\n**重要提示（必须遵守）**：如果用户确认已经购买了机票，但在此处没有订单信息，**必须引导用户选择订单**。")

        qaList = args.get("qaList", "")
        if qaList:
            sections.append(f"#### 行李相关知识问答对:\n{qaList}")
        luggageBizKnowledge = args.get("luggageBizKnowledge", "")
        if luggageBizKnowledge:
            sections.append(f"#### 行李业务知识:\n{luggageBizKnowledge}")

        return {
            "context": "\n\n".join(sections),
            # "confirm": "亲，您已经购买了成人票吗？",
            # "orderStatusNew": None,
            "needConfirm": False
        }

    # 1. 解析订单信息
    if order_detail_str:
        try:
            orderDetail = json.loads(order_detail_str)
            canWrongPurchaseRefund = '否'
            try:
                # 当前可选择的退票原因列表处理
                return_reasons = orderDetail.get('returnReasons', [])  # 增加默认空列表
                if return_reasons:  # 检查return_reasons是否为空
                    data = json.loads(return_reasons)
                    for item in data:
                        if isinstance(item, dict) and item.get('code') == 54 and item.get('needUpload') is False:
                            canWrongPurchaseRefund = '是'
            except (json.JSONDecodeError, TypeError) as e:
                canWrongPurchaseRefund = '否'

            # 构建订单信息字符串的各行
            order_info_lines = [
                f"{{订单状态:{orderDetail.get('orderStatusNew')}}}",
                f"{{是否有儿童:{orderDetail.get('is_have_children')}}}",
                f"{{成人票价:{orderDetail.get('adult_sale_price')}}}",
                f"{{成人数量:{orderDetail.get('adult_count')}}}",
                f"{{成人姓名:{orderDetail.get('adult_name')}}}",
                f"{{儿童票价:{orderDetail.get('child_sale_price')}}}",
                f"{{儿童数量:{orderDetail.get('child_count')}}}",
                f"{{儿童姓名:{orderDetail.get('child_name')}}}",
                f"{{婴儿票价:{orderDetail.get('infant_sale_price')}}}",
                f"{{婴儿数量:{orderDetail.get('infant_count')}}}",
                f"{{婴儿姓名:{orderDetail.get('infant_name')}}}",
                f"{{行程类型:{orderDetail.get('trip_type')}}}",
                f"{{支付时间:{orderDetail.get('pay_time')}}}",
                f"{{第一程航班号:{orderDetail.get('first_flight_no')}}}",
                f"{{第二程航班号:{orderDetail.get('second_flight_no')}}}",
                f"{{第一程起飞时间:{orderDetail.get('first_depart_time')}}}",
                f"{{第二程起飞时间:{orderDetail.get('second_depart_time')}}}",

                f"{{第一程实际起飞时间:{orderDetail.get('firstActualDepartTime')}}}",
                f"{{第二程实际起飞时间:{orderDetail.get('secondActualDepartTime')}}}",

                f"{{第一程预计到达时间:{orderDetail.get('firstArriveTime')}}}",
                f"{{第二程预计到达时间:{orderDetail.get('secondArriveTime')}}}",

                f"{{第一程实际到达时间:{orderDetail.get('firstActualArriveTime')}}}",
                f"{{第二程实际到达时间:{orderDetail.get('secondActualArriveTime')}}}",

                f"{{第一程值机柜台:{orderDetail.get('firstCkiCounter')}}}",
                f"{{第二程值机柜台:{orderDetail.get('secondCkiCounter')}}}",

                f"{{第一程登机口:{orderDetail.get('firstDeptGate')}}}",
                f"{{第二程到达口:{orderDetail.get('secondDeptGate')}}}",

                f"{{第一程到达口:{orderDetail.get('firstDestExit')}}}",
                f"{{第二程航班号:{orderDetail.get('second_flight_no')}}}",

                f"{{第一程行李转盘:{orderDetail.get('firstCarousel')}}}",
                f"{{第二程行李转盘:{orderDetail.get('secondCarousel')}}}",

                f"{{第一程成人舱位代码:{orderDetail.get('firstAdultCabin')}}}",
                f"{{第二程成人舱位代码:{orderDetail.get('secondAdultCabin')}}}",

                f"{{第一程准点率:{orderDetail.get('firstPunctualityRate')}}}",
                f"{{第二程准点率:{orderDetail.get('secondPunctualityRate')}}}",

                f"{{第一程餐食信息:{orderDetail.get('firstMealDesc')}}}",
                f"{{第二程餐食信息:{orderDetail.get('secondMealDesc')}}}",

                f"{{第一程免费手提行李额信息:{orderDetail.get('firstFreeHandBaggage')}}}",
                f"{{第二程免费手提行李额信息:{orderDetail.get('secondFreeHandBaggage')}}}",

                f"{{第一程免费托运行李额:{orderDetail.get('firstFreeCheckinBaggage')}}}",
                f"{{第二程免费托运行李额:{orderDetail.get('secondFreeCheckinBaggage')}}}",

                f"{{第一程实际承运航司:{orderDetail.get('firstOperateCarrier')}}}",
                f"{{第一程实际承运航司:{orderDetail.get('secondOperateCarrier')}}}",

                f"{{第一程航司电话:{orderDetail.get('firstCarrierPhone')}}}",
                f"{{第二程航司电话:{orderDetail.get('secondCarrierPhone')}}}",

                f"{{出发城市:{orderDetail.get('depart_city_name')}}}",
                f"{{出发机场:{orderDetail.get('depart_airport_name')}}}",
                f"{{出发机场代码:{orderDetail.get('depart_airport_code')}}}",
                f"{{到达城市:{orderDetail.get('arrive_city_name')}}}",
                f"{{到达机场:{orderDetail.get('arrive_airport_name')}}}",
                f"{{到达机场代码:{orderDetail.get('arrive_airport_code')}}}",
                f"{{中转城市:{orderDetail.get('transit_city_name')}}}",
                f"{{中转机场:{orderDetail.get('transit_airport_name')}}}",
                f"{{中转机场代码:{orderDetail.get('transit_airport_code')}}}",
                f"{{美团在线客服是否在工作时间:{orderDetail.get('isWorking')}}}",
                f"{{美团在线客服工作开始时间:{orderDetail.get('workTimeBegin')}}}",
                f"{{美团在线客服工作结束时间:{orderDetail.get('workTimeEnd')}}}",
                f"{{当前距离出票时间的小时数:{orderDetail.get('ticketIssuedHour')}}}",
                f"{{当前是否可以申请错购退:{canWrongPurchaseRefund}}}",
                f"{{退票费用和金额:{orderDetail.get('refundFeeAndAmount')}}}",
                f"{{未申请退票时候自愿退预计退款完成时间:{orderDetail.get('unRefundVolMoneyRefundTime')}}}",
                f"{{票号描述:{orderDetail.get('ticketNoDesc')}}}"
            ]

            # 过滤掉值为None或空字符串的行
            order_info = "\n".join([line for line in order_info_lines if
                                    ':' in line and line.split(':')[1].rstrip('}') not in ['None', '']])

            # 将订单信息添加到上下文
            # order_info_text = "\n".join(order_info)
            sections.append(f"#### 当前用户订单信息:\n{order_info}")
            # 提取订单信息
            depart_city = orderDetail.get("depart_city_name")
            arrive_city = orderDetail.get("arrive_city_name")
            depart_time = orderDetail.get("first_depart_time")

            # 订单存在，输出确认文字
            confirm_text = f"请问您是要咨询{depart_time}从{depart_city}出发，到达{arrive_city}的订单吗？"

        except Exception as e:
            # 出现异常时返回错误信息
            return {
                "orderInfo": "",
                "errorMsg": f"生成订单信息时出错: {str(e)}"
            }

    # 2. 添加知识库内容
    qaList = args.get("qaList", "")
    if qaList:
        sections.append(f"#### 行李相关知识问答对:\n{qaList}")
    luggageBizKnowledge = args.get("luggageBizKnowledge", "")
    if luggageBizKnowledge:
        sections.append(f"#### 行李业务知识:\n{luggageBizKnowledge}")

    # 3. 拼接上下文
    context_text = "\n\n".join(sections)

    return {
        "context": context_text,
        "confirm": confirm_text,
        # "currentTime": current_time,
        "needConfirm": True if confirm_text else False
    }



if __name__ == "__main__":
    input = {
      "orderId": "96225111113502836388939",
      "orderDetail": "{\"depart_airport_code\":\"PEK\",\"firstArriveTime\":\"2025-11-13 22:20:00\",\"secondDeptGate\":\"\",\"first_depart_time\":\"2025-11-13 20:40:00\",\"adult_name\":\"张淑娟，张德禧\",\"transit_airport_name\":\"\",\"child_name\":\"\",\"depart_city_name\":\"北京\",\"arrive_city_name\":\"南京\",\"second_depart_time\":\"\",\"refundFeeAndAmount\":\"当前时段退票总手续费：【416元/人】，成人退票手续费为：【208元/人】，儿童退票手续费：【0元/人】，婴儿退票手续费：【0元/人】，预计退款金额：756元\",\"second_flight_no\":\"\",\"secondCkiCounter\":\"\",\"child_sale_price\":\"0\",\"secondPunctualityRate\":\"\",\"firstCkiCounter\":\"J,K\",\"child_count\":\"0\",\"secondMealDesc\":\"\",\"secondDestExit\":\"\",\"transit_airport_code\":\"\",\"pay_time\":\"2025-11-11 13:51:53\",\"firstDeptGate\":\"\",\"carrier_mobile\":\"95583\",\"ticketIssuedHour\":\"1.4\",\"depart_airport_name\":\"首都机场\",\"firstPunctualityRate\":\"91.8%\",\"adult_sale_price\":\"516\",\"orderStatusNew\":\"出票完成\",\"secondFreeHandBaggage\":\"\",\"firstActualDepartTime\":\"\",\"cxr_phone\":\"中国国航:95583\",\"arrive_airport_code\":\"NKG\",\"is_have_children\":\"否\",\"trip_type\":\"单程\",\"secondCarrierPhone\":\"\",\"ticketNoDesc\":\"张德禧：<br>北京-南京 票号：9992537161606<br>张淑娟：<br>北京-南京 票号：9992537161607\",\"firstMealDesc\":\"无餐食\",\"firstOperateCarrier\":\"\",\"unRefundVolMoneyRefundTime\":\"2025-11-11 19:29:51\",\"infant_count\":\"0\",\"returnReasons\":\"[{\\\"code\\\": 37, \\\"reason\\\": \\\"取消行程/不想飞了/订错票了\\\", \\\"returnType\\\": 1, \\\"needUpload\\\": false}, {\\\"code\\\": 54, \\\"reason\\\": \\\"2小时购错票退票政策\\\", \\\"returnType\\\": 2, \\\"needUpload\\\": false}, {\\\"code\\\": 33, \\\"reason\\\": \\\"航班取消/航班时刻变更\\\", \\\"returnType\\\": 2, \\\"needUpload\\\": true}, {\\\"code\\\": 43, \\\"reason\\\": \\\"航司/机场拒载/其他\\\", \\\"returnType\\\": 2, \\\"needUpload\\\": true}, {\\\"code\\\": 34, \\\"reason\\\": \\\"身体原因\\\", \\\"returnType\\\": 2, \\\"needUpload\\\": true}]\",\"firstCarousel\":\"\",\"secondActualArriveTime\":\"\",\"secondArriveTime\":\"\",\"arrive_airport_name\":\"禄口机场\",\"firstActualArriveTime\":\"\",\"secondFreeCheckinBaggage\":\"\",\"freeCheckinSpecial\":\"\",\"first_flight_no\":\"CA1819\",\"secondAdultCabin\":\"\",\"firstFreeHandBaggage\":\"限重5KG，单件体积不超过40×20×55cm，可带1件/人\",\"secondActualDepartTime\":\"\",\"secondCarousel\":\"\",\"transit_city_name\":\"\",\"firstFreeCheckinBaggage\":\"限重20KG，单件体积不超过60×40×100cm\",\"firstAdultCabin\":\"K\",\"infant_name\":\"\",\"firstCarrierPhone\":\"95583\",\"orderTag\":\"[81,49,70,90]\",\"adult_count\":\"2\",\"infant_sale_price\":\"0\",\"firstDestExit\":\"\"}",
      "qaList": "[{\"score\":0.88,\"content\":\"问题:为什么我没有免费行李额?\",\"answer\":\"问题:为什么我没有免费行李额? 回答:亲，部分舱位、部分机票时不带免费行李额的，免费行李额信息在下单页面会有提示，具体以下单时的行李额说明为准。\"},{\"score\":0.88,\"content\":\"问题:免费行李额标准是什么\",\"answer\":\"问题:免费行李额标准是什么\\n回答:您好，机票免费行李额标准因舱位、航司及航线而异，建议购票前可在航班产品说明处查看行李额规则，避免临时超重费用。\"},{\"score\":0.87,\"content\":\"问题:超重行李托运费报销\",\"answer\":\"问题:超重行李托运费报销\\n回答:您好，美团平台暂不支持加购托运行李，如需单独购买或加购，咨询费用问题，您必须要联系航空公司购买。行李购买的报销凭证，如您需要联系航空公司索要。\"},{\"score\":0.87,\"content\":\"问题:咨询可托运/携带物品\",\"answer\":\"问题:咨询可托运/携带物品\\n回答:一般物品随身携带上机/托运规则\\n液体托运：小于等于100毫升的液体（化妆品、牙膏等）可以随身携带；大于100毫升的必须托运。\\n充电宝：随身携带不超过2个，总容量小于等于160wh（32000毫安），单个充电宝小于等于20000毫安。\\n禁止随身携带任何酒精、打火机、火柴和管制刀具乘机。\\n温馨提示：\\n更多关于其他物品是否可携带，可以联系航空公司咨询。\"},{\"score\":0.86,\"content\":\"问题:购买行李额/托运多少钱？\",\"answer\":\"问题:购买行李额/托运多少钱？ 回答:亲，因为不同航司、不同产品计算方式都不同，我们这里无法得知购买行李额/托运的具体金额\"}]",
      "luggageBizKnowledge": "问题:婴儿物品携带规定\n回答:奶粉：有婴儿随行，可携带适量婴儿奶粉/牛奶/母乳等。\n手推车：每个婴儿可免费托运婴儿手推车一辆。\n玩具：婴儿玩具如果不是刀枪类型的违禁品玩具，正常也可以携带。\n问题:水果生鲜携带规定 回答:水果生鲜可托运也可随身携带，随身携带时要注意不要超过行李重量。 水果：可以携带包装完好无异味的水果上机；不得携带异味、液体水分的水果，如榴莲、椰子等。 肉类：可以携带真空包装完好无异味的肉类上机。 蔬菜：可以携带包装完好无异味的蔬菜上机。 海鲜：可以携带真空包装完好无异味的海鲜上机；不得携带活体，冰冻的海鲜上机（冰冻海鲜包装完好可办理托运。） 蛋类：鸡蛋属于易碎品，在运输中容易出现破损影响到其他行李，大部分航司不允许带上飞机、不建议托运。罐头/食品/炒菜：罐头或食品中可能含有液体，所以建议密封包装完好进行托运。\n问题:充电宝携带规定\n回答:坐飞机可携带1个额定能量小于100WH（20000毫安）的充电宝，最多可随身携带不超过2个，加在一起的总容量不超过160WH（32000毫安），且每个都要有明确的品牌和规格。如果单个充电宝在20000-32000毫安之间，需要拨打航司电话咨询，具体请以航空公司为准。\n温馨提示：\n民航局发布通知，自6月28日起禁止旅客携带没有3C标识、3C标识不清晰、被召回型号或批次的充电宝乘坐境内航班，具体召回信息可在“国家市场监督管理总局缺陷产品召回技术中心”官方网站→“消费品召回”栏目查询，请您注意检查携带充电宝是否符合规定哦。\n问题:化妆品/饮料等液体携带规定\n回答:一般航空公司允许携带少量旅行自用化妆品，如口红、眼影盘、睫毛膏、眉笔、眼线笔等固体可以随身携带。液体类如洗面奶、眼霜、乳液、牙膏、洗发水、面膜等，其容器容积总和不得超过100毫升，并应置于独立袋内，接受开瓶检查。\n如需携带更多化妆品及其他液体（譬如食用油、洗衣液等），请办理托运。\n问题:药品携带规定\n回答:如果旅客携带液体、凝胶以及喷雾类液态药物，须附有医生处方或医院证明方可豁免，而药物数量则以旅客在飞机上的所需为准，而对容器和塑料袋并无要求。携带液态药物须分开过检，向安检人员出示证明。详细情况需拨打航空公司电话，转人工咨询。\n问题:烟酒携带规定\n回答:每个乘客最多可随身携带两条香烟，托运香烟则是无限制。\n酒禁止随身携带，可以托运，托运规则如下：\n托运时需标识全面清晰、零售外包装完好，每个容器体积不得超过5L。\n酒精度在24至70之间，每位旅客托运数量不能超过5L。\n酒精度超过70的不可托运。\n散装白酒是不能托运。\n问题:宠物携带规定\n回答:小动物属于民航限制运输的物品行列，旅客不能随身携带进入客舱，但是可以作为行李托运或者作为货物运输。一般航空公司对每架航班携带小动物的数量都有限制，美团无法核实，您必须提前与航空公司确认，经航空公司同意后方可托运。\n问题:数码产品电器携带规定\n回答:内含锂电池的数码产品和电器（如手表、手提电脑、摄像机、电熨斗等），及备用电池，可以放在手提行李里带上客舱。\n设备中电池及备用电池锂含量和额定能量限制如下:\n锂金属或锂合金电池锂含量不超过2g。\n备用电池的锂含量或额定能量不超过2g或100Wh的，航空公司允许旅客携带不超过（含）2块备用电池，以旅客在行程中使用设备所需的合理数量为判断标准。\n锂电池额定能量在大于100Wh但不超过160Wh的，经航空有公司批准后,每人携带不超过2块。\n问题:乐器及画具携带规定\n回答:您好，类似吉他、小提琴、颜料等特殊易损坏物品，您需要联系实际托运航空公司咨询，以免起飞当天航司拦截耽误您的行程。\n问题:常见拉杆箱尺寸说明\n回答:以下是常见的行李箱尺寸长高宽（cm）。可以作为参考，但是不同品牌的行李箱规格可能不同，所以需要用户确认自己行李箱的长高宽。16寸（31*43*13）、18寸（34*44*20）、20寸（34*50*20）、22寸（39*58*24）、24寸（42*68*26）、26寸（45*72*28）、28寸（47*78*28）、32寸（53*88*30）",
      "workTimeBegin":"06:00",
      "workTimeEnd": "23:00",
      "isWorking": "1",
    }
    print(main(input))