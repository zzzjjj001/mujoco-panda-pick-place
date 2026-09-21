"""抓取任务的失败类型诊断。"""


def classify_failure(ep) -> str | None:
    """根据单 episode 统计归类失败原因; 成功返回 None。

    ep 需要包含: grasp_success, place_success, max_lift, contact_ever,
    min_tcp_dist, cube_off_table。
    """
    if ep["grasp_success"] and ep["place_success"]:
        return None
    if ep["cube_off_table"]:
        return "KNOCKED_OFF_TABLE"   # 把立方体碰落桌面
    if ep["grasp_success"]:
        return "MISSED_BIN"          # 抓起并抬升成功，但没放进放置区
    if ep["max_lift"] >= 0.05:
        return "SLIPPED"             # 一度抬起但中途滑落
    if ep["contact_ever"]:
        return "GRASP_FAILED"        # 接触了立方体但没能稳定夹住
    if ep["min_tcp_dist"] > 0.05:
        return "NOT_REACHED"         # 末端从未接近立方体
    return "GRASP_FAILED"            # 靠近了但没有发生接触
