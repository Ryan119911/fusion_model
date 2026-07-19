# Fusion Model：整字直接生成

`main` 分支保留原来的单笔 B-BSMG，同时新增纯 U-Net 整字模型：

```text
完整轨迹空间图：骨架/压力/笔顺/方向 cos/方向 sin
        ↓
U-Net 编码器 + 多尺度跳跃连接 + U-Net 解码器
        ↓
一个解码器一次输出完整 128×128 字图
```

新流程不会先预测每一笔再叠加，也不使用 Transformer 或全局 token pooling。旧的
`bbsmg_best.pt` 仍是单笔 checkpoint；整字流程使用独立的 `character_best.pt`。

详细命令见 [docs/whole_character.md](docs/whole_character.md)。
