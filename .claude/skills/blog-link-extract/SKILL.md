---
name: blog-link-extract
description: This skill should be used when processing a Chinese blog post markdown file to extract inline markdown links from the body and consolidate them into a "链接汇总" (link summary) section at the end of the file.
version: 1.0.0
---

# Blog Link Extract Skill

将中文博客正文中的 markdown 内联链接提取至文末"链接汇总"部分。

## 操作步骤

1. **读取目标文件**，识别正文中所有 `[文字](URL)` 格式的内联链接。

2. **替换正文中的链接**：将每处 `[文字](URL)` 替换为纯文字（仅保留链接文本，去掉括号和 URL）。

3. **在文件末尾追加"链接汇总"部分**，格式如下：

```markdown
## 链接汇总

- 描述性标签: [https://example.com](https://example.com)
- 描述性标签: [https://example.com/page](https://example.com/page)
```

## 规则

- 按链接在正文中出现的顺序排列。
- 每条链接的标签使用语义化的中文描述，而非原始链接文字（例如"发布说明"而非"点击这里"）。
- 代码块（` ```...``` `）内的链接不处理。
- 纯 URL（不含 markdown 格式，如 `https://...`）保持原样，不移动。
- 相对路径链接（如 `/docs`、`/tutorials`）不移动到链接汇总，直接去掉链接格式保留文字。
- 若文件已有"链接汇总"部分，则追加新链接，不重复已有条目。
