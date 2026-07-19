# Fusion Model：整字直接生成

`main` 分支从 `014c0f1` 继续开发，保留原来的单笔 B-BSMG，同时新增真正的整字模型：

```text
完整字的全部笔画 10D 特征序列
        ↓
共享笔画编码器 + 笔顺位置编码 + Transformer
        ↓
一个解码器一次输出完整 128×128 字图
```

新流程不会先预测每一笔再叠加。旧的 `bbsmg_best.pt` 仍是单笔 checkpoint；整字流程使用独立的 `character_best.pt`。

详细命令见 [docs/whole_character.md](docs/whole_character.md)。
