# m08_HiKEY_hybrid_light

`m08_HiKEY_hybrid_light` 是一个轻量版 HiKEY 检索方法。
它保留 m05 的“题干 + 选项多路检索”，但不再做 m07 那种章节硬路由，而是改成“本地结构加权重排”。

## 工作流程

```text
题干 / 选项多路召回
-> HiKEY evidence pack
-> 本地结构打分
-> 多路去重与重排
-> 单次最终作答
```

## 主要信号

- `field_card`、`financial_field`、`table_row` 这类结构类型
- `section_path` 的层级深度
- 公司、年份、指标名的精确匹配
- 领域关键词
- 泛化内容惩罚

## 和 m07 的区别

m07 是先路由章节，再在少数章节里检索。
m08 不做硬路由，而是直接在 HiKEY 缓存里召回，再做本地重排。

所以它更像“多路召回 + 本地重排”，不是传统的纯向量/关键词混合检索。

## 召回诊断

m08 会输出 `retrieval_diagnostics.json`，里面能看到：

- 每题 top-k 中哪些证据命中了 gold 选项
- 哪些 gold 选项被覆盖
- 哪些 gold 选项缺失
- 按领域汇总的命中率

这套统计和 m07 一样，属于弱监督 proxy recall，不是人工标注级别的绝对召回。
