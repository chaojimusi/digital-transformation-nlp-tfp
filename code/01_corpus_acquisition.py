# -*- coding: utf-8 -*-
"""
01_corpus_acquisition.py —— 第一步：语料获取（Corpus Acquisition）
==================================================================
对应论文"3.1 样本选择与数据来源"。

关键动作：定源、对齐、留痕
    * 定源    —— 语料来源为上市公司年报"管理层讨论与分析（MD&A）"章节
    * 对齐    —— 文本年度 t 与财务变量年度 t 严格对齐（年报于 t+1 年 4 月披露，
                 本文按"会计年度"对齐，并在稳健性检验中采用 t-1 期文本）
    * 留痕    —— 保存原始文件、记录来源与处理脚本，保证可复现

⚠️ 数据说明（重要）
------------------
本仓库用于**方法演示与可复现性验证**。受公开授权限制，仓库中的 MD&A 语料为
**按真实语料的统计特征合成的模拟语料**（synthetic corpus），财务变量同为模拟值。
模拟过程使用固定随机种子，完全可复现；在真实研究中，只需将
`data/raw/mda_corpus.csv` 替换为从巨潮资讯网 / CNRDS 下载并经 PDF 解析的真实
MD&A 文本，并将 `data/raw/firm_financials.csv` 替换为 CSMAR 导出的财务面板，
其余五步流水线与代码**无需修改**即可在原样本上复现。
详见 docs/数据来源说明.md。

输出：
    data/raw/mda_corpus.csv          语料库（公司-年度 + 文本）
    data/raw/firm_financials.csv     财务与公司治理面板（CSMAR 位置）
    data/raw/_simulation_truth.csv   模拟真值（仅用于校验流水线，不进入分析）
    data/raw/annual_reports_sample/  样例年报文本（*.txt）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib

cfg = importlib.import_module("00_config")

rng = np.random.default_rng(cfg.SEED)

# ------------------------------------------------------------------
# 1. 公司主表：行业 / 地区 / 产权 / 高新技术资质
# ------------------------------------------------------------------
INDUSTRIES = [
    "电子设备制造", "机械设备制造", "医药生物", "基础化工", "汽车制造",
    "家用电器", "食品饮料", "信息传输与软件服务", "批发和零售", "交通运输",
    "电力热力", "土木工程建筑",
]
# 行业的技术密集度系数（决定数字化潜变量的行业差异）
IND_SECTOR_EFFECT = {
    "信息传输与软件服务": 0.70, "电子设备制造": 0.45, "机械设备制造": 0.20,
    "医药生物": 0.15, "汽车制造": 0.15, "家用电器": 0.10, "基础化工": 0.05,
    "批发和零售": 0.10, "交通运输": 0.05, "电力热力": 0.00,
    "食品饮料": -0.05, "土木工程建筑": -0.10,
}
REGIONS = ["东部", "中部", "西部"]
REGION_P = [0.62, 0.22, 0.16]

SURNAME = list("华晟中联瑞宏恒远天启东方金泰创科开元众合盛昌隆鑫嘉信德力通达")
MIDWORD = ["", "", "海", "达", "通", "远", "创", "盛"]

firms = []
for i in range(1, cfg.N_FIRMS + 1):
    industry = INDUSTRIES[rng.integers(0, len(INDUSTRIES))]
    region = rng.choice(REGIONS, p=REGION_P)
    soe = int(rng.random() < 0.32)
    hitech = int(rng.random() < (0.55 if industry in ("信息传输与软件服务", "电子设备制造", "医药生物") else 0.28))
    name = f"{rng.choice(SURNAME)}{rng.choice(MIDWORD)}{rng.choice(SURNAME)}{rng.choice(['科技','股份','实业','电子','智能','材料','生物','能源'])}股份有限公司"
    firms.append({
        "firm_id": f"F{i:04d}",
        "stock_code": f"{int(rng.integers(1, 700)):03d}{int(rng.integers(0, 1000)):03d}",
        "firm_name": name,
        "industry": industry,
        "region": region,
        "SOE": soe,
        "HiTech": hitech,
        "list_age_0": int(rng.integers(1, 26)),   # 2017 年时的上市年限
    })
firms = pd.DataFrame(firms)
firms["firm_effect"] = rng.normal(0, 1, len(firms))          # 公司不可观测异质性
firms["digital_firm_effect"] = rng.normal(0, 1, len(firms))

# ------------------------------------------------------------------
# 2. 生成潜变量（数字化转型的真实强度）——仅用于生成语料
# ------------------------------------------------------------------
# 行业-年度共同冲击：同行业企业当年共同面临的数字化"浪潮"
# （它只通过企业数字化程度影响 TFP，从而为工具变量法提供外生变异）
IND_YEAR_SHOCK = {
    f"{ind}|{yr}": rng.normal()
    for ind in INDUSTRIES for yr in cfg.YEARS
}

rows = []
for _, f in firms.iterrows():
    for t, year in enumerate(cfg.YEARS):
        digital_latent = (
            1.05 * f["digital_firm_effect"]
            + 0.34 * t
            + IND_SECTOR_EFFECT[f["industry"]]
            + 0.55 * IND_YEAR_SHOCK[f"{f['industry']}|{year}"]
            + 0.30 * f["HiTech"]
            + 0.20 * (f["region"] == "东部")
            - 0.10 * f["SOE"]
            + 0.12 * rng.normal()
        )
        rows.append({"firm_id": f["firm_id"], "year": year,
                     "digital_latent": digital_latent, "t_index": t})
panel_latent = pd.DataFrame(rows)
panel_latent["dig_std"] = (panel_latent["digital_latent"] - panel_latent["digital_latent"].mean()) / panel_latent["digital_latent"].std()
panel_latent = panel_latent.merge(firms, on="firm_id", how="left")

# ------------------------------------------------------------------
# 3. 词典文本生成器：把潜变量"翻译"为可被词典命中的中文文本
# ------------------------------------------------------------------
digital_dict = pd.read_csv(cfg.PATH_DIGITAL_DICT, encoding="utf-8")
WORDS_A = digital_dict.loc[digital_dict["dimension"] == "A", "word"].tolist()   # 底层技术运用
WORDS_B = digital_dict.loc[digital_dict["dimension"] == "B", "word"].tolist()   # 技术实践应用
WORDS_ALL = WORDS_A + WORDS_B

# 通用（非数字化）句子模板：刻意不包含任何词典词
GENERIC_TEMPLATES = [
    "报告期内，公司主营业务保持总体稳定，主要产品产销量较上年有所增长。",
    "公司持续优化产品结构，中高端产品收入占比进一步提升。",
    "面对复杂多变的宏观环境，公司坚持稳中求进，经营质量稳步改善。",
    "报告期内，公司加强成本费用管控，综合毛利率较上年略有提高。",
    "公司进一步完善内部治理机制，规范运作水平持续提升。",
    "公司主要客户结构保持稳定，前五大客户销售占比与上年基本持平。",
    "报告期内，公司完成多项产线技术改造，产能利用率处于合理区间。",
    "公司继续加大人才培养力度，员工队伍结构得到优化。",
    "公司持续关注原材料价格波动，通过集中采购降低采购成本。",
    "报告期内，公司获得多项发明专利与实用新型专利授权。",
    "公司积极拓展海外市场，出口收入较上年同期有所增长。",
    "公司资产负债结构保持稳健，流动比率处于行业合理水平。",
    "报告期内，公司完成了一次限制性股票激励计划的首期授予。",
    "公司高度重视安全生产与环境保护，报告期内未发生重大安全事故。",
    "公司持续完善质量管理体系，主要产品一次合格率保持稳定。",
    "报告期内，公司经营性现金流量净额较上年明显改善。",
    "公司董事会下设审计、薪酬与考核等专门委员会，治理结构不断完善。",
    "公司所处行业竞争格局总体稳定，集中度较上年略有上升。",
    "报告期内，公司新建产能陆续投产，规模效应逐步显现。",
    "公司持续加强品牌建设，品牌影响力在细分领域进一步提升。",
    "公司主要产品的平均销售价格较上年基本稳定。",
    "报告期内，公司研发人员数量较上年有所增加。",         # 注意：不含词典词
    "公司持续强化存货与应收账款管理，营运效率有所提高。",
    "公司积极履行社会责任，持续参与定点帮扶与公益捐赠。",
    "报告期内，公司不存在被监管部门处罚的情形。",
    "公司主要原材料供应渠道畅通，供应链稳定性良好。",
    "公司销售费用率较上年小幅下降，费用投放效率提高。",
    "报告期内，公司完成对下属子公司少数股权的收购。",
    # —— 以下为进一步扩充的经营表述，用于提升合成语料的词汇多样性 ——
    "公司主营业务收入较上年同期增长，主要产品市场份额稳中有升。",
    "报告期内，公司持续推进降本增效，单位生产成本较上年有所下降。",
    "公司聚焦主业，逐步收缩非核心业务，资源配置效率得到提升。",
    "公司坚持以客户为中心，持续完善售前咨询与售后服务体系。",
    "报告期内，公司新增授权专利若干项，技术储备进一步充实。",
    "公司加强与高校及科研院所合作，产学研协同创新持续推进。",
    "公司主要产品通过了多项行业认证，市场准入门槛进一步打开。",
    "报告期内，公司加强应收账款回收管理，回款周期有所缩短。",
    "公司持续优化供应商结构，关键原材料采购价格保持稳定。",
    "公司销售网络覆盖全国主要区域，渠道下沉取得积极进展。",
    "报告期内，公司在多个区域新增销售网点，市场覆盖范围扩大。",
    "公司强化预算管理，期间费用率较上年同期有所下降。",
    "公司不断完善信息披露机制，投资者关系管理水平稳步提高。",
    "报告期内，公司完成董事会换届，治理架构保持稳定。",
    "公司严格执行环保标准，报告期内未发生环保违法行为。",
    "公司持续开展员工职业技能培训，人才梯队建设成效显现。",
    "报告期内，公司主要产品的产销率保持在较高水平。",
    "公司加强与国际客户的合作，海外订单数量较上年增加。",
    "公司密切关注汇率变动，通过套期保值降低汇兑风险。",
    "报告期内，公司募集资金项目按计划推进，投资进度符合预期。",
    "公司健全内部控制体系，风险防范能力进一步增强。",
    "公司主要生产设备的综合利用率维持在较高水平。",
    "报告期内，公司完成部分老旧设备的更新改造。",
    "公司持续推进节能降耗，单位产品能耗较上年有所下降。",
    "公司积极参与行业标准制定，行业影响力进一步提升。",
    "报告期内，公司主要原材料的采购渠道保持稳定。",
    "公司加强安全生产培训，员工安全意识明显增强。",
    "公司持续完善薪酬考核机制，员工积极性得到有效调动。",
    "报告期内，公司完成了多项管理制度的修订与完善。",
    "公司积极应对市场竞争，产品价格策略保持稳健。",
    "公司主要子公司的经营业绩较上年有所改善。",
    "报告期内，公司研究开发支出较上年同期有所增加。",
    "公司持续加大技改投入，生产自动化水平稳步提升。",
    "公司主营业务毛利率较上年基本持平，盈利能力保持稳定。",
    "报告期内，公司完成了年度经营计划的主要目标。",
    "公司持续优化组织架构，管理效率进一步提高。",
    "公司加强合同评审与履约管理，法律风险得到有效控制。",
    "报告期内，公司获得地方政府多项政策支持。",
    "公司主要产品在细分市场保持较强的竞争优势。",
    "公司持续关注下游需求变化，及时调整产品结构。",
    "报告期内，公司完成了对部分生产线的技术升级。",
    "公司加强与金融机构的合作，融资渠道保持畅通。",
    "公司严格执行关联交易决策程序，交易定价公允。",
    "报告期内，公司员工总数较上年略有增加。",
    "公司持续健全风险预警机制，经营风险总体可控。",
    "公司主要产品的售后服务满意度较上年有所提升。",
    "报告期内，公司完成了质量管理体系的年度审核。",
    "公司积极履行纳税义务，报告期内依法足额缴纳税款。",
]

DIGITAL_TEMPLATES = [
    "公司持续推进{dw}建设，相关项目已完成阶段性验收。",
    "报告期内，公司围绕{dw}开展专项工作，取得积极成效。",
    "公司加大{dw}投入力度，相关资源保障充足。",
    "依托{dw}，公司持续优化业务流程与决策效率。",
    "公司积极布局{dw}，力争形成新的竞争壁垒。",
    "在{dw}方面，公司已完成整体规划并进入实施阶段。",
    "公司成立由主要负责人牵头的{dw}工作组，统筹推进相关工作。",
    "报告期内，公司引入{dw}解决方案，运营效率明显提升。",
    "公司以{dw}为抓手，推动业务模式创新。",
    "公司持续深化{dw}应用，客户体验得到改善。",
    "公司计划进一步扩大{dw}的应用范围。",
    "围绕{dw}，公司建立了常态化推进机制。",
    "公司通过{dw}实现了对生产环节的精细化管理。",
    "报告期内，公司在{dw}领域的投入较上年同期增长明显。",
    "公司将{dw}纳入年度重点工作，明确了责任部门与时间表。",
    "公司借助{dw}提升了供应链协同水平。",
    "公司加快推进{dw}，相关能力建设取得进展。",
    "报告期内，公司完成{dw}相关系统的上线运行。",
    "公司以{dw}为支撑，持续提升管理精细化程度。",
    "公司结合自身业务特点，稳步推进{dw}落地。",
    "公司组建专业技术团队，负责{dw}的选型与实施。",
    "报告期内，公司在{dw}方向的投入较上年明显增加。",
    "公司以{dw}为切入点，重构了现有的业务流程。",
    "公司通过引入{dw}，有效降低了人工操作差错率。",
    "公司围绕{dw}制定了三年行动方案并启动实施。",
    "报告期内，公司{dw}项目的建设进度符合预期。",
    "公司在{dw}方面与外部专业机构开展了深度合作。",
]

def build_text(z: float, n_sent: int) -> str:
    """根据标准化潜变量 z 生成一份 MD&A 文本。"""
    p_digi = min(0.58, max(0.02, cfg.DIGI_BASE_P + cfg.DIGI_SLOPE * (1 / (1 + np.exp(-1.35 * z)))))
    n_digi = int(rng.binomial(n_sent, p_digi))
    n_gen = n_sent - n_digi
    sents = list(rng.choice(GENERIC_TEMPLATES, size=n_gen, replace=True))
    for _ in range(n_digi):
        tpl = DIGITAL_TEMPLATES[rng.integers(0, len(DIGITAL_TEMPLATES))]
        k = int(rng.integers(cfg.HITS_PER_SENT[0], cfg.HITS_PER_SENT[1] + 1))
        # 高潜变量公司更倾向使用"底层技术运用"类词汇
        p_a = min(0.75, max(0.20, 0.38 + 0.16 * z))
        words = []
        for _k in range(k):
            pool = WORDS_A if rng.random() < p_a else WORDS_B
            words.append(pool[rng.integers(0, len(pool))])
        sent = tpl
        first = True
        for w in words:
            if first:
                sent = sent.replace("{dw}", w, 1); first = False
            else:
                sent = sent.replace("{dw}", w, 1) if "{dw}" in sent else sent + f"公司同时推进{w}相关工作。"
        sent = sent.replace("{dw}", words[0])
        sents.append(sent)
    rng.shuffle(sents)
    # 补一句财务口径的"硬信息"，模拟年报中普遍存在的数字表述
    header = f"报告期内，公司实现营业收入{int(rng.integers(30000, 900000))}万元，归属于上市公司股东的净利润{int(rng.integers(800, 90000))}万元。"
    return header + "".join(sents)

texts = []
for _, r in panel_latent.iterrows():
    n_sent = int(rng.integers(cfg.SENT_MIN, cfg.SENT_MAX + 1))
    texts.append(build_text(r["dig_std"], n_sent))
panel_latent["text"] = texts

corpus = panel_latent[[
    "firm_id", "stock_code", "firm_name", "year", "industry", "region", "SOE", "HiTech", "text"
]].copy()
corpus.to_csv(cfg.PATH_CORPUS, index=False, encoding="utf-8-sig")
print(f"[步骤1] 语料库已生成：{cfg.PATH_CORPUS.name}  观测 {len(corpus)} 条，"
      f"平均文本长度 {corpus['text'].str.len().mean():.0f} 字")

# ------------------------------------------------------------------
# 4. 财务与治理面板（对应 CSMAR 导出表）
# ------------------------------------------------------------------
fin = panel_latent[["firm_id", "year", "dig_std", "firm_effect", "t_index"]].copy()
n = len(fin)

fin["Size"] = 22.00 + 0.55 * rng.normal(size=n) + 0.06 * fin["t_index"]
fin["Lev"] = np.clip(0.42 + 0.15 * rng.normal(size=n), 0.05, 0.95)
fin["ROA"] = 0.045 + 0.032 * rng.normal(size=n)
fin["Growth"] = 0.12 + 0.26 * rng.normal(size=n)
fin["Age"] = np.log(1 + np.clip(panel_latent["list_age_0"].values + panel_latent["t_index"].values, 1, 40))
fin["Top1"] = np.clip(0.34 + 0.14 * rng.normal(size=n), 0.05, 0.80)
fin["Dual"] = (rng.random(n) < 0.25).astype(int)
fin["Board"] = np.log(np.clip(np.round(5 + 6 * rng.random(n)), 5, 15))

# —— 效度检验所需的辅助变量（企业创新投入与"数字化痕迹"）——
fin["RD"] = np.clip(3.60 + 1.15 * fin["dig_std"] + 0.90 * panel_latent["HiTech"].values
                    + 0.25 * (fin["Size"] - 22.0) + 0.60 * rng.normal(size=n), 0.0, 30.0)
fin["Intangible"] = np.clip(0.055 + 0.025 * fin["dig_std"] + 0.020 * rng.normal(size=n), 0.0, 0.5)
fin["ITexec"] = np.clip(0.12 + 0.055 * fin["dig_std"] + 0.050 * rng.normal(size=n), 0.0, 1.0)
fin["DigitalInvest"] = np.clip(0.018 + 0.011 * fin["dig_std"] + 0.008 * rng.normal(size=n), 0.0, 0.3)

# —— 被解释变量：全要素生产率（模拟 CSMAR 的 LP / OP 估算结果）——
BETA_DIG_TRUE = 0.075          # 数字化的"真实"直接效应（对标准化潜变量）
BETA_HITECH_EXTRA = 0.100      # 高新技术企业数字化的额外效应（异质性来源）
TFP_LP = (
    8.30
    + BETA_DIG_TRUE * fin["dig_std"]
    + BETA_HITECH_EXTRA * fin["dig_std"] * panel_latent["HiTech"].values
    + 0.020 * fin["RD"]                                  # 创新投入的中介渠道
    + 0.045 * panel_latent["HiTech"].values
    - 0.035 * panel_latent["SOE"].values
    + 0.030 * (panel_latent["region"].values == "东部")
    + 0.100 * (fin["Size"] - 22.0)
    - 0.250 * (fin["Lev"] - 0.42)
    + 3.000 * (fin["ROA"] - 0.045)
    - 0.050 * fin["Age"]
    + 0.45 * fin["firm_effect"]
    + 0.34 * rng.normal(size=n)
)
fin["TFP_LP"] = TFP_LP
fin["TFP_OP"] = 0.92 * TFP_LP + 0.30 * rng.normal(size=n) + 0.66

fin = fin.merge(panel_latent[["firm_id", "year", "industry", "region", "SOE", "HiTech"]],
                on=["firm_id", "year"], how="left")
fin = fin[["firm_id", "year", "industry", "region", "SOE", "HiTech",
           "Size", "Lev", "ROA", "Growth", "Age", "Top1", "Dual", "Board",
           "RD", "Intangible", "ITexec", "DigitalInvest", "TFP_LP", "TFP_OP"]]
fin.to_csv(cfg.DIR_RAW / "firm_financials.csv", index=False, encoding="utf-8-sig")
print(f"[步骤1] 财务面板已生成：firm_financials.csv  观测 {len(fin)} 条")

# ------------------------------------------------------------------
# 5. 模拟真值（仅用于校验：衡量测度的"回收率"，不进入论文分析）
# ------------------------------------------------------------------
truth = panel_latent[["firm_id", "year", "dig_std"]].copy()
truth.to_csv(cfg.DIR_RAW / "_simulation_truth.csv", index=False, encoding="utf-8-sig")

# ------------------------------------------------------------------
# 6. 导出样例年报文本，便于人工查看
# ------------------------------------------------------------------
smp = corpus.sample(cfg.TEXT_SAMPLE_N, random_state=cfg.SEED)
for _, r in smp.iterrows():
    fp = cfg.DIR_SAMPLE / f"{r['firm_id']}_{r['year']}_MD&A.txt"
    fp.write_text(
        f"【样例】公司：{r['firm_name']}｜股票代码：{r['stock_code']}｜年度：{r['year']}\n"
        f"【章节】管理层讨论与分析（MD&A）\n"
        f"【说明】本文件为按真实语料统计特征合成的模拟语料，仅用于方法演示。\n"
        + "-" * 60 + "\n" + r["text"],
        encoding="utf-8")
print(f"[步骤1] 已导出 {len(smp)} 份样例年报文本至 {cfg.DIR_SAMPLE.name}/")
