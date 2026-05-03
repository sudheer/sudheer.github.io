---
title: Searching Specific File Types in Emacs
description: Search only selected file extensions using consult-ripgrep.
pubDate: 2025-03-23
tags:
  - tools
category: Micro
draft: false
---

If you're using Emacs with `consult-ripgrep` (`M-x consult-ripgrep`), you might want to search only within certain file types—say, just `.scala` files in a project.

```text
M-x consult-ripgrep RET -- -g '*.scala'
```

- The `--` separates ripgrep (`rg`) options from the search term.
- `-g '*.scala'` tells `rg` to search only in `.scala` files.
- You can add multiple `-g` flags for different file types:

```text
M-x consult-ripgrep RET -- -g '*.scala' -g '*.sbt'
```
