"""一次性构建 申万一级 code→行业 全市场映射缓存（31次调用覆盖所有A股）。
申万成分接口按IP时间冷却(~每几分钟放3次)，故分块(每块3个)+块间长睡+可续传(已映射行业跳过)。fire-and-forget。"""
import os
import time
import pandas as pd
import akshare as ak

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, "sector_map_full.csv")
PARTIAL = os.path.join(_HERE, "sector_map_partial.csv")


def load_partial():
    if os.path.exists(PARTIAL):
        m = pd.read_csv(PARTIAL, dtype={"code": str})
        return {str(c).zfill(6): s for c, s in zip(m["code"], m["sector"])}
    return {}


def save(code2sec, path):
    pd.DataFrame(sorted(code2sec.items()), columns=["code", "sector"]).to_csv(
        path, index=False, encoding="utf-8-sig")


def main():
    first = ak.sw_index_first_info()
    code2sec = load_partial()
    done_secs = set(code2sec.values())
    todo = [(str(r["行业代码"]).split(".")[0], r["行业名称"])
            for _, r in first.iterrows() if r["行业名称"] not in done_secs]
    print(f"申万一级31个，已完成{len(done_secs)}，待拉{len(todo)}", flush=True)
    i = 0
    while i < len(todo):
        chunk = todo[i:i + 3]
        for sym, name in chunk:
            ok = False
            for _ in range(2):
                try:
                    cons = ak.index_component_sw(symbol=sym)
                    if "证券代码" in cons.columns and len(cons):
                        for cc in cons["证券代码"].astype(str):
                            code2sec[cc.zfill(6)] = name
                        ok = True
                        break
                except Exception:
                    pass
                time.sleep(8)
            print(f"  {name:<8} {'OK' if ok else '跳过(限流)'}  累计 {len(code2sec)} 只/{len(set(code2sec.values()))} 行业", flush=True)
            if ok:
                save(code2sec, PARTIAL)
            time.sleep(2)
        i += 3
        if i < len(todo):
            time.sleep(240)            # 块间冷却
    if len(set(code2sec.values())) >= 25:
        save(code2sec, OUT)
        print(f"已缓存 {OUT}：{len(code2sec)} 只 / {len(set(code2sec.values()))} 行业", flush=True)
    else:
        print(f"覆盖不足({len(set(code2sec.values()))}行业)，仅存partial。重跑本脚本续传。", flush=True)


if __name__ == "__main__":
    main()
