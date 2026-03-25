Fetch the blog post at the following URL, convert it to markdown, and save both an English and Chinese version.

URL: $ARGUMENTS

## Steps

1. Use WebFetch to retrieve the full content of the URL. Extract the complete blog post including title, date, author, all sections, code blocks, and lists. Format it as clean markdown, do not change the blog content, just make it format as markdown, each section of markdown do not use "---" as the separator, ignore "Learning Resources" section in the footer of webpage which is not part of the blog content.

2. Determine the filename from the URL slug (the last path segment, e.g. `pytorch-2-11-release-blog` from `https://pytorch.org/blog/pytorch-2-11-release-blog/`).

3. Save the English markdown to `posts/<filename>.md`.

4. Translate the entire markdown content into Chinese. Preserve all markdown formatting, code blocks (only translate comments inside code, not code itself), and all hyperlinks (translate link text but keep URLs unchanged).

5. Save the Chinese translation to `post-cn/<filename>.md`.

6. Apply the `blog-link-extract` skill to `post-cn/<filename>.md`: extract all inline markdown links from the body, replace them with plain text, and append a "链接汇总" section at the end of the file.

Report the two saved file paths when done.
