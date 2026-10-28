# kacem-portfolio

Personal site and blog. Plain HTML and CSS generated from a config file and Markdown posts. One dependency (`marked`).

## Run it

```sh
npm install
npm run dev      # builds with drafts, serves at http://localhost:3000
npm run build    # production build into dist/ (drafts excluded)
```

## Edit the home page

Everything on the home page is in `site.config.mjs`: the focus sentence, intro, research, experience, projects, and links. Leave `research` empty to hide that section.

## Write a post

1. Copy `content/posts/_template.md` to `content/posts/my-post.md`.
   If the post has images, use a folder instead: `content/posts/my-post/index.md`, with the images next to it.
2. Fill in `title`, `date` (YYYY-MM-DD), and `summary`. Keep `draft: true` while writing.
3. Preview with `npm run dev`.
4. Delete the `draft: true` line, commit, and push. Vercel rebuilds the site.

Math uses `$...$` or `\(...\)` inline and `$$...$$` or `\[...\]` for display, rendered with KaTeX. Files starting with `_` are ignored.

## Deploy

`vercel.json` tells Vercel to run `npm run build` and serve `dist/`. The output is static, so GitHub Pages or Netlify work too.
