# 新 FULL15 对 E2 独立单训 DFlash：正式长测

2026-10-11新FULL15正式对照：用户明确将短测改为长测；E2独立单训DFlash对新FULL15_NOFILL联合模型。状态complete，正式窗口8/8。C1 T0 AL+13.86% TPS+9.94% 隔离True；C1 T1 AL+18.19% TPS+14.09% 隔离True；C40 T0 AL+13.70% TPS+6.98% 隔离True；C40 T1 AL+18.75% TPS+12.04% 隔离True。三层头/权重未改，仅沿用已核验的局部取消调试预填充；不测尾部截断，不重训；不同主体权重的整体模型对比。

GSM8K nonthinking；C1=200、C40=1319；max_new=2096/context4096；T0/T1；BF16、batch-invariant target、CUDA Graph；block16=anchor+15 proposals。GPU7空卡依次运行，先各自数值smoke再正式。C1 KV52GiB，C40正式KV44GiB、smoke38GiB，各对照臂一致。历史正式C40用46GiB；本报告只作本次同设置直接比较。

AC按此前接收长度AL展示：AL=1+accepted/drafts，包含anchor/bonus；proposal接受率另存COMPARISON.json。单轮正式面板，不是独立训练重复。

| 设置 | 单训 AL | 新 FULL15 AL | AL 增长 | 单训 TPS | 新 FULL15 TPS | TPS 增长 | 隔离 |
|---|---:|---:|---:|---:|---:|---:|---|
| C1 T0 | 7.478576 | 8.515070 | +13.86% | 178.410 | 196.152 | +9.94% | True |
| C1 T1 | 6.335933 | 7.488577 | +18.19% | 151.859 | 173.253 | +14.09% | True |
| C40 T0 | 7.609462 | 8.652091 | +13.70% | 3146.145 | 3365.701 | +6.98% | True |
| C40 T1 | 6.482782 | 7.698110 | +18.75% | 2687.595 | 3011.228 | +12.04% | True |

代码/权重身份见[native_protocol.json](native_protocol.json)与[PREFLIGHT.json](PREFLIGHT.json)。独立body SHA0b8cef16…8ce7，联合body SHAad4ab70f…8168，头SHA6635f429…944f；沿用已核验iter26/27 runtime，开发选用iter20/21不覆盖。

即时日志：[C1 smoke](single_card_C1_smoke.log)、[C1正式](single_card_C1_formal.log)、[C40 smoke](single_card_C40_smoke.log)、[C40正式](single_card_C40_formal.log)。[状态](status.json)、[原始比较数据](COMPARISON.json)。
