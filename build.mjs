// Builds the site into dist/. Usage: node build.mjs [--drafts]
// Posts live in content/posts/ as either `slug.md` or `slug/index.md` (with images beside it).
// Files whose name starts with "_" are ignored.

import fs from "node:fs";
import path from "node:path";
import { marked } from "marked";
import site from "./site.config.mjs";

const ROOT = path.dirname(new URL(import.meta.url).pathname);
const OUT = path.join(ROOT, "dist");
const POSTS_DIR = path.join(ROOT, "content", "posts");
const INCLUDE_DRAFTS = process.argv.includes("--drafts");
const KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist";

// ---------- helpers ----------

const esc = (s) =>
  String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

const inline = (s) => marked.parseInline(s ?? "");

function write(rel, html) {
  const file = path.join(OUT, rel);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, html);
}

function copyDir(src, dest, skip = () => false) {
  if (!fs.existsSync(src)) return;
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    if (skip(entry.name)) continue;
    const from = path.join(src, entry.name);
    const to = path.join(dest, entry.name);
    if (entry.isDirectory()) copyDir(from, to, skip);
    else {
      fs.mkdirSync(dest, { recursive: true });
      fs.copyFileSync(from, to);
    }
  }
}

// Front matter: simple `key: value` lines between --- fences.
function parseFrontMatter(raw) {
  const m = raw.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?/);
  if (!m) return { data: {}, body: raw };
  const data = {};
  for (const line of m[1].split(/\r?\n/)) {
    const kv = line.match(/^(\w+):\s*(.*)$/);
    if (!kv) continue;
    let v = kv[2].trim().replace(/^["']|["']$/g, "");
    if (v === "true") v = true;
    else if (v === "false") v = false;
    data[kv[1]] = v;
  }
  return { data, body: raw.slice(m[0].length) };
}

// Pull math out before Markdown runs (so _ and \ survive), then put it back.
// Supports $$...$$ and \[...\] (display), $...$ and \(...\) (inline). Code is left alone.
function renderMarkdown(src) {
  const math = [];
  const pattern =
    /(```[\s\S]*?```|`[^`\n]+`)|\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)|(?<![\\$\w])\$(?!\s)([^$\n]+?)(?<!\s)\$(?![\d$])/g;
  const protectedSrc = src.replace(pattern, (whole, code, d1, d2, i1, i2) => {
    if (code) return code;
    const display = d1 ?? d2;
    const tex = display ?? i1 ?? i2;
    math.push({ tex: tex.trim(), display: display !== undefined });
    return `MATHTOKEN${math.length - 1}X`;
  });
  let html = marked.parse(protectedSrc);
  html = html.replace(/<p>MATHTOKEN(\d+)X<\/p>/g, (_, i) => `MATHTOKEN${i}X`);
  html = html.replace(/MATHTOKEN(\d+)X/g, (_, i) => {
    const { tex, display } = math[i];
    return display
      ? `<div class="math-display">\\[${esc(tex)}\\]</div>`
      : `<span class="math-inline">\\(${esc(tex)}\\)</span>`;
  });
  return { html, hasMath: math.length > 0 };
}

// ---------- page shell ----------

function page({ title, description, body, path: urlPath = "/", math = false, type = "website" }) {
  const fullTitle = title ? `${title} | ${site.name}` : site.name;
  const mathHead = math
    ? `
  <link rel="stylesheet" href="${KATEX}/katex.min.css">
  <script defer src="${KATEX}/katex.min.js"></script>
  <script defer src="${KATEX}/contrib/auto-render.min.js"
    onload="renderMathInElement(document.querySelector('article'), {delimiters: [{left: '\\\\[', right: '\\\\]', display: true}, {left: '\\\\(', right: '\\\\)', display: false}], throwOnError: false})"></script>`
    : "";
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${esc(fullTitle)}</title>
  <meta name="description" content="${esc(description || site.description)}">
  <meta property="og:title" content="${esc(title || site.name)}">
  <meta property="og:description" content="${esc(description || site.description)}">
  <meta property="og:type" content="${type}">
  <meta name="color-scheme" content="light dark">
  <link rel="icon" href="/favicon.ico">
  <link rel="stylesheet" href="/style.css">${mathHead}
  <script defer src="/_vercel/insights/script.js"></script>
  <script defer src="/_vercel/speed-insights/script.js"></script>
</head>
<body>
  <header class="site-header">
    <a class="site-name" href="/">${esc(site.name)}</a>
    <nav aria-label="Main">
      <a href="/writing/"${urlPath.startsWith("/writing") ? ' aria-current="page"' : ""}>Writing</a>
    </nav>
  </header>
  <main>
${body}
  </main>
  <footer class="site-footer">
    <a href="mailto:${esc(site.email)}">${esc(site.email)}</a>
    ${site.links.map((l) => `<a href="${esc(l.url)}">${esc(l.label)}</a>`).join("\n    ")}
  </footer>
</body>
</html>
`;
}

// ---------- posts ----------

function loadPosts() {
  if (!fs.existsSync(POSTS_DIR)) return [];
  const posts = [];
  for (const entry of fs.readdirSync(POSTS_DIR, { withFileTypes: true })) {
    if (entry.name.startsWith("_") || entry.name.startsWith(".")) continue;
    let file, assetsDir, slug;
    if (entry.isDirectory()) {
      file = path.join(POSTS_DIR, entry.name, "index.md");
      if (!fs.existsSync(file)) continue;
      assetsDir = path.join(POSTS_DIR, entry.name);
      slug = entry.name;
    } else if (entry.name.endsWith(".md")) {
      file = path.join(POSTS_DIR, entry.name);
      slug = entry.name.replace(/\.md$/, "");
    } else continue;

    const { data, body } = parseFrontMatter(fs.readFileSync(file, "utf8"));
    if (!data.title) throw new Error(`${path.relative(ROOT, file)} needs a "title" in its front matter`);
    if (data.draft && !INCLUDE_DRAFTS) continue;
    const tags = String(data.tags ?? "")
      .split(",")
      .map((t) => t.trim())
      .filter(Boolean);
    const order = data.order === undefined || data.order === "" ? Infinity : Number(data.order);
    posts.push({ ...data, tags, order, slug, body, assetsDir });
  }
  // Lower `order` first; posts without one go last, alphabetically.
  return posts.sort((a, b) => a.order - b.order || a.title.localeCompare(b.title));
}

const tagSlug = (t) =>
  t
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");

function collectTags(posts) {
  const tags = new Map();
  for (const p of posts) for (const t of p.tags) tags.set(t, [...(tags.get(t) ?? []), p]);
  return [...tags.entries()].sort((a, b) => b[1].length - a[1].length || a[0].localeCompare(b[0]));
}

const tagLinks = (tags) =>
  tags.length
    ? `<p class="tags">${tags.map((t) => `<a href="/writing/tags/${tagSlug(t)}/">${esc(t)}</a>`).join(" ")}</p>`
    : "";

function postList(posts, current) {
  return `<ul class="post-list">
${posts
  .map((p) => {
    const title =
      current && p.slug === current
        ? `<span class="current">${esc(p.title)}</span>`
        : `<a href="/writing/${p.slug}/">${esc(p.title)}</a>`;
    const draft = p.draft ? ' <span class="draft">draft</span>' : "";
    const summary = !current && p.summary ? `\n        <p>${inline(p.summary)}</p>` : "";
    return `      <li>\n        ${title}${draft}${summary}\n      </li>`;
  })
  .join("\n")}
    </ul>`;
}

// ---------- pages ----------

function homePage(posts) {
  const sections = [];

  sections.push(`    <p class="focus">${inline(site.focus)}</p>
    <p>${inline(site.intro)}</p>`);

  if (site.research?.length) {
    sections.push(`    <section aria-labelledby="research">
      <h2 id="research">Research</h2>
      <ul class="entries">
${site.research
  .map(
    (r) => `        <li>
          ${r.url ? `<a href="${esc(r.url)}">${esc(r.title)}</a>` : esc(r.title)}${r.venue ? ` <span class="venue">${inline(r.venue)}</span>` : ""}
          ${r.summary ? `<p>${inline(r.summary)}</p>` : ""}
        </li>`
  )
  .join("\n")}
      </ul>
    </section>`);
  }

  if (posts.length) {
    sections.push(`    <section aria-labelledby="writing">
      <h2 id="writing">Writing</h2>
      ${postList(posts.slice(0, 5))}${posts.length > 5 ? `\n      <p><a href="/writing/">All ${posts.length} posts</a></p>` : ""}
    </section>`);
  }

  if (site.experience?.length) sections.push(`    <section aria-labelledby="experience">
      <h2 id="experience">Experience</h2>
      <dl class="cv">
${site.experience
  .map((e) => `        <dt>${esc(e.when)}</dt>\n        <dd><span class="org">${esc(e.org)}</span> ${esc(e.role)}</dd>`)
  .join("\n")}
      </dl>
    </section>`);

  if (site.projects?.length) {
    sections.push(`    <section aria-labelledby="projects">
      <h2 id="projects">Projects</h2>
      <ul class="entries">
${site.projects
  .map(
    (p) => `        <li>
          ${p.url ? `<a href="${esc(p.url)}">${esc(p.title)}</a>` : esc(p.title)}
          <p>${inline(p.summary)}</p>
        </li>`
  )
  .join("\n")}
      </ul>
    </section>`);
  }

  if (site.outside) sections.push(`    <p class="outside">${inline(site.outside)}</p>`);

  return page({ body: sections.join("\n\n"), path: "/" });
}

function writingIndex(posts, tags) {
  if (!posts.length) {
    return page({
      title: "Writing",
      body: `    <h1>Writing</h1>\n    <p class="muted">No posts yet.</p>`,
      path: "/writing/",
    });
  }
  const nav = tags.length
    ? `\n    <nav class="tag-nav" aria-label="Topics">${tags
        .map(([t, ps]) => `<a href="/writing/tags/${tagSlug(t)}/">${esc(t)} <span class="count">${ps.length}</span></a>`)
        .join(" ")}</nav>`
    : "";
  return page({
    title: "Writing",
    description: `Writing by ${site.name}`,
    body: `    <h1>Writing</h1>${nav}\n    ${postList(posts)}`,
    path: "/writing/",
  });
}

function tagPage(tag, posts) {
  return page({
    title: tag,
    description: `Posts about ${tag.toLowerCase()} by ${site.name}`,
    body: `    <h1>${esc(tag)}</h1>
    <p class="muted">${posts.length} post${posts.length === 1 ? "" : "s"}. <a href="/writing/">All writing</a></p>
    ${postList(posts)}`,
    path: `/writing/tags/${tagSlug(tag)}/`,
  });
}

function postPage(post, tagMap) {
  const { html, hasMath } = renderMarkdown(post.body);
  const firstTag = post.tags[0];
  const related = firstTag ? tagMap.get(firstTag) : null;
  const more =
    related && related.length > 1
      ? `
    <section class="related" aria-labelledby="related">
      <h2 id="related">More on ${esc(firstTag.toLowerCase())}</h2>
      ${postList(related, post.slug)}
    </section>`
      : "";
  const body = `    <article>
      <header class="post-header">
        <h1>${esc(post.title)}${post.draft ? ' <span class="draft">draft</span>' : ""}</h1>
        ${tagLinks(post.tags)}
      </header>
${html}
    </article>${more}
    <p class="back"><a href="/writing/">All writing</a></p>`;
  return page({
    title: post.title,
    description: post.summary,
    body,
    path: `/writing/${post.slug}/`,
    math: hasMath,
    type: "article",
  });
}

function notFound() {
  return page({
    title: "Page not found",
    body: `    <h1>Page not found</h1>\n    <p>This page doesn't exist. Try the <a href="/">home page</a> or the <a href="/writing/">writing index</a>.</p>`,
  });
}

// ---------- build ----------

fs.rmSync(OUT, { recursive: true, force: true });
copyDir(path.join(ROOT, "public"), OUT);

const posts = loadPosts();
const tags = collectTags(posts);
const tagMap = new Map(tags);
write("index.html", homePage(posts));
write("writing/index.html", writingIndex(posts, tags));
write("404.html", notFound());
for (const [tag, tagged] of tags) write(`writing/tags/${tagSlug(tag)}/index.html`, tagPage(tag, tagged));
for (const post of posts) {
  write(`writing/${post.slug}/index.html`, postPage(post, tagMap));
  if (post.assetsDir) {
    copyDir(post.assetsDir, path.join(OUT, "writing", post.slug), (name) => name === "index.md");
  }
}

console.log(
  `Built ${posts.length} post${posts.length === 1 ? "" : "s"} and ${tags.length} tag page${tags.length === 1 ? "" : "s"} into dist/${INCLUDE_DRAFTS ? " (drafts included)" : ""}`
);
