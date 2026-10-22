# Long-Short Tax Saving：V2

旧 A/B/C 版本保存在 GitHub 的 [v1.0.0](https://github.com/asd1354403003/Long-Short-Tax-Saving/tree/v1.0.0)。本地已迁出仍需复用的下载和协整工具，删除旧版入口、调仓代码及两份 CSV。当前开发分支 `research/pair-pool-v2` 暂不推送。

## 运行顺序

1. `pair_pool.ipynb`：2022–2024 筛选、拟合和训练 Z，2025 验证，按验证 Sharpe 降序保存 `data/pair_pool.csv`。
2. `analysis.ipynb`：读取该表，获取 2026 测试期 Open/Close，在内存中计算持仓和盈亏，不额外保存 CSV 或价格缓存。

训练逻辑在 `pair_training.py`，组合回测在 `pair_backtest.py`，共用下载/统计包装在 `market_data.py`，EG/ADF 在 `cointegration.py`，市场 beta 与 VaR/CVaR 在 `risk.py`。使用相对路径。

## 资金与开平仓

- 默认 `max_pairs=15`、`long_amount=1_000_000`。每组 long 的名义预算为 100 万/15，约 $66,666.67；short 同样设此上限。按 beta 统一缩放，使两条腿各自不超过上限，再向下取整为整数股，所以实际金额可能更低。限额在开仓时检查，之后固定股数，价格变化可以使敞口漂移，不每天 trim。
- 只使用验证通过且 refit 有效的 pair。使用截至 2025 年底估计的 refit_alpha、refit_beta、refit_mu_e、refit_sigma_eq，在整个 2026 固定，不重新训练；entry_z/exit_z 来自此前训练。
- 从 2026-01-01 开始，但实际成交在首个有数据的交易日。可用 2025 最后一个交易日 Close 生成首日信号。按验证 Sharpe 从高到低扫描，达到开仓阈值才开：z 高时 short Y/long X，z 低时反向。Close 信号在下一交易日 Open 成交。
- z 回到或穿过平仓阈值，或单笔按初始总敞口计算的亏损达到 12%，则下一 Open 平掉两条腿。跳空可以使实际亏损超过止损值。
- 先执行平仓，再按排名补充未交易过的 pair，最多同时 15 组。每组 pair 一旦实际开仓，平掉后不再使用；未成交的不算用过。默认同一 ticker 不在多组同时持仓里重复，但平仓后可以参与另一组未使用的 pair；可通过 allow_shared_tickers 修改。
- 无信号、标的冲突或候选耗尽时就留现金，不强行补满。active_pairs / used_pairs / unused_pairs 显示名额和候选消耗。
- 缺少开仓 Open 就跳过该候选；缺少平仓 Open 就保留待执行退出指令。缺少 Close 用此前已知估值，新仓用开仓价，记录 stale_price_legs，不用未来价格回填。
- 最后一天不强制清仓，保留未实现盈亏。训练为了比较完整期间盈亏而做的期末强平，不用于本次 2026 持仓策略。end_date=None 时，纽约时间 17:00 后才纳入当天价格，否则截止此前交易日。

## 原来的 Z 方法

不使用滚动窗口。价差 `e=Close_y-alpha-beta*Close_x`，训练期拟合 `e_t=C+B*e_(t-1)+噪声`，要求 `0<B<1`。保持原 mean_square 公式：

```text
tau=1/252
theta=-log(B)/tau
mu_e=C/(1-B)
sigma_eq=sqrt(mean(e**2)*tau/(1-exp(-2*theta*tau)))
z=(e-mu_e)/sigma_eq
```

测试 Z=0.25～3.5、间隔 0.25；默认 exit_z=Z/2。至少 5 笔训练交易，先按训练累计预算收益，再按 Sharpe 选 Z。2025 冻结模型验证，至少 2 笔完整交易且训练/验证盈亏和 Sharpe 为正才通过。另以 2022–2025 重估 refit 参数，供 2026 使用，不改变验证和 Z 选择。2026 不参与拟合、调参或候选筛选。

## 输出与盈亏

Notebook 展示每日多空金额、与 SPY 的累计收益比较、仓位数量、开平仓事件、当前持仓、已实现/未实现盈亏、市场 beta 和历史/正态 VaR、CVaR。事件包含信号日期与实际成交日期。

long 股数/市值为正，short 为负。固定参考账户预算为 200 万美元，包含闲置现金：每日收益=每日 P&L/200 万，累计收益=累计 P&L/200 万，不将这类每日预算收益连乘。买卖资金流不是利润，隔夜跳空和新仓开盘到收盘的盈亏均计入。

始终有：累计 P&L=累计已实现 P&L+当前未实现 P&L。已实现 loss 按每条腿的负盈亏相加，即使整组盈利，其中亏损的一条腿仍进入 realized_loss；gain 则单独加总正盈亏。

只用 Open/Close，不用 Adj Close；交易费、借券、融资、分红现金流和现金利息暂不计，SPY 也只计价格收益。市值/AUM、行业标签及 ETF 目录使用当前快照，有幸存者及筛选偏差。股票门槛为当前市值 100 亿美元，ETF AUM 1 亿美元，训练期日均成交额 2,000 万美元；相关性 >0.90，EG 95%、残差 ADF -2.86、lag 1，不降低门槛凑数。

## 税务范围

当前计算交易盈亏和已实现亏损，不计算实际抵税额或税后财富。协整或更换 ticker 不保证符合 wash sale/straddle；商品 ETP 的税务结构还需另行处理。

- Wash sale：[26 U.S.C. §1091](https://uscode.house.gov/us/usc/119-118/t26/s1091/) 使用 “substantially identical stock or securities”；一般涉及卖出前后各 30 天。换组合名称不能消除证券层面的限制。
- Straddle：[26 U.S.C. §1092](https://uscode.house.gov/us/usc/119-118/t26/s1092/) 针对相互抵消风险的头寸。亏损可因对冲头寸的未实现收益而递延，下一年仍可能继续递延，跨年不会自动解禁。
- [IRS Publication 550](https://www.irs.gov/publications/p550) 提供 wash sales、straddles 及协调规则。英文 README 保留官方原文摘录与递延示例。
