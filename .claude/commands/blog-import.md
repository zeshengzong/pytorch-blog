Fetch the blog post at the following URL, convert it to markdown, and save both an English and Chinese version.

URL: $ARGUMENTS

## Critical Rule: Verbatim Extraction

**You are a transcriber, NOT a writer.** Your job is to faithfully reproduce the exact text from the webpage. You must NEVER summarize, paraphrase, rewrite, reorganize, add section headings that don't exist in the original, or omit any content. Every sentence in the output must be traceable to a sentence on the webpage.

## Steps

### Step 1: Extract raw blog content from the webpage

Use **two methods** to extract the blog content, then cross-reference them:

**Method A** (preferred): Use the Chrome MCP tool `get_page_text` to extract the full raw text from the page. This gives you the original unprocessed text.
- First call `tabs_context_mcp` (with createIfEmpty=true) to get a tab
- Then `navigate` to the URL
- Then call `get_page_text` to extract the raw text

**Method B** (fallback, only if Method A fails): Use `WebFetch` with this exact prompt:
```
Act as a text extractor only. Return the COMPLETE text content of this blog post exactly as written on the page. Copy every paragraph, heading, quote, bullet point, and link verbatim. Do NOT summarize, paraphrase, reorganize, add new section titles, or omit any text. Output the raw text in the same order it appears on the page.
```

### Step 2: Convert raw text to clean markdown

Take the raw text from Step 1 and format it as markdown following these strict rules:

- **Preserve all original headings exactly as they appear** on the page. Do NOT invent new section headings like "Overview", "Key Highlights", or "Supporting Quotes" if they don't exist in the original.
- **Preserve every paragraph verbatim.** Do not merge, split, rephrase, or summarize paragraphs.
- **Preserve all quotes exactly** with `>` blockquote formatting.
- **Preserve all bullet points and numbered lists exactly** as written.
- **Preserve all hyperlinks** in `[text](url)` format.
- **Preserve all code blocks** with proper ``` fencing.
- Do NOT use `---` horizontal rules as section separators.
- Ignore the "Learning Resources" section in the footer — it is not part of the blog content.
- Include the blog title as `# Title`, and include date/author metadata if visible on the page.

### Step 3: Self-check before saving

Before saving, verify:
- Does every heading in your markdown exist on the original webpage? If not, remove it.
- Does every paragraph in your markdown match a paragraph on the webpage word-for-word? If not, fix it.
- Have you omitted any paragraphs that exist on the webpage? If so, add them back.
- Is the content in the same order as the original? If not, reorder it.

### Step 4: Save the English version

Determine the filename from the URL slug (the last path segment, e.g. `pytorch-2-11-release-blog` from `https://pytorch.org/blog/pytorch-2-11-release-blog/`).

Save the English markdown to `posts/<filename>.md`.

### Step 5: Translate to Chinese

Translate the entire markdown content into Chinese. Rules:
- Translate sentence by sentence. Do NOT summarize or change any sentence structure while translating.
- Preserve all markdown formatting, code blocks (only translate comments inside code, not code itself), and all hyperlinks (translate link text but keep URLs unchanged).

Save the Chinese translation to `post-cn/<filename>.md`.

### Step 6: Extract links from Chinese version

Apply the `blog-link-extract` skill to `post-cn/<filename>.md`: extract all inline markdown links from the body, replace them with plain text, and append a "链接汇总" section at the end of the file, each link write as key-value format as following, do not use markdown format link. Link example:
- PyTorch Foundation: https://pytorch.org/foundation/

Report the two saved file paths when done.
