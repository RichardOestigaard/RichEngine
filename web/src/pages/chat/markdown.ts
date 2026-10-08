// Ported verbatim from server/chat.html: escapeHtml, inlineMd, renderMarkdown,
// codeBlock, mdStableEnd, displayContent, plus the paintContent two-tier scheme
// as createMarkdownPainter.

export function escapeHtml(text: string): string {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

// Inline runs after escaping, so only tags this emits can appear.
export function inlineMd(text: string): string {
  return escapeHtml(text)
    .replace(/`([^`\n]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/~~([^~\n]+)~~/g, "<del>$1</del>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
}

// Minimal markdown: fenced code, headings, lists (incl. task items),
// tables, quotes, paragraphs. An unclosed fence mid-stream renders as an
// open code block, which is what the reader sees anyway.
export function renderMarkdown(text: string): string {
  const out: string[] = [];
  let para: string[] = [];
  let list: { type: "ul" | "ol"; items: { text: string; checked?: boolean }[] } | null = null;
  let code: { lang: string; lines: string[] } | null = null;
  const flushPara = () => {
    if (para.length) out.push(`<p>${para.map(inlineMd).join("<br>")}</p>`);
    para = [];
  };
  const flushList = () => {
    if (list)
      out.push(
        `<${list.type}>${list.items
          .map(item =>
            item.checked === undefined
              ? `<li>${inlineMd(item.text)}</li>`
              : `<li class="task"><input type="checkbox" disabled${item.checked ? " checked" : ""}> ${inlineMd(item.text)}</li>`
          )
          .join("")}</${list.type}>`
      );
    list = null;
  };
  const flushAll = () => { flushPara(); flushList(); };
  const cells = (line: string) =>
    line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map(c => c.trim());
  const isSep = (line: string) =>
    line.includes("|") &&
    cells(line).every(c => /^:?-+:?$/.test(c) || c === "");
  const lines = text.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (code !== null) {
      if (/^```\s*$/.test(line)) {
        out.push(codeBlock(code));
        code = null;
      } else code.lines.push(line);
      continue;
    }
    const fence = /^```(\w*)\s*$/.exec(line);
    const heading = /^(#{1,4})\s+(.*)/.exec(line);
    const task = /^\s*[-*•]\s+\[([ xX])\]\s+(.*)/.exec(line);
    const bullet = /^\s*[-*•]\s+(.*)/.exec(line);
    const numbered = /^\s*\d+[.)]\s+(.*)/.exec(line);
    const quote = /^>\s?(.*)/.exec(line);
    if (fence) { flushAll(); code = { lang: fence[1] || "", lines: [] }; }
    else if (heading) {
      flushAll();
      const level = Math.min(heading[1].length + 2, 6);
      out.push(`<h${level}>${inlineMd(heading[2])}</h${level}>`);
    }
    // A table row is a header line of | cells followed by a --- separator;
    // body rows follow until the first line without a pipe.
    else if (
      line.includes("|") && i + 1 < lines.length && isSep(lines[i + 1])
    ) {
      flushAll();
      const header = cells(line);
      const rows: string[][] = [];
      for (i++; i + 1 < lines.length && lines[i + 1].includes("|"); i++)
        if (lines[i + 1].trim()) rows.push(cells(lines[i + 1]));
      out.push(
        `<div class="tablewrap"><table><thead><tr>${header.map(c => `<th>${inlineMd(c)}</th>`).join("")}</tr></thead>` +
        `<tbody>${rows.map(r => `<tr>${r.map(c => `<td>${inlineMd(c)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`
      );
    }
    else if (task || bullet || numbered) {
      flushPara();
      const type: "ul" | "ol" = numbered ? "ol" : "ul";
      if (!list || list.type !== type) { flushList(); list = { type, items: [] }; }
      list.items.push(
        task
          ? { text: task[2], checked: task[1] !== " " }
          : { text: (bullet || numbered)![1] }
      );
    }
    else if (quote) { flushAll(); out.push(`<blockquote>${inlineMd(quote[1])}</blockquote>`); }
    else if (!line.trim()) flushAll();
    else para.push(line);
  }
  flushAll();
  if (code !== null) out.push(codeBlock(code));
  return out.join("");
}

// The fence language (or "text") labels the header; copy reads the code back.
function codeBlock(code: { lang: string; lines: string[] }): string {
  const lang = escapeHtml(code.lang || "text");
  return `<div class="codeblock"><div class="code-head"><span>${lang}</span>` +
    `<button class="copy" type="button">Copy</button></div>` +
    `<pre><code>${escapeHtml(code.lines.join("\n"))}</code></pre></div>`;
}

// The position after the last blank line outside a code fence: everything
// before it renders to finished blocks, everything after it may still grow.
export function mdStableEnd(text: string): number {
  let end = 0, fence = false, offset = 0;
  for (const line of text.split("\n")) {
    if (/^```/.test(line)) fence = !fence;
    // A blank line ends a block only where a newline follows it; the last
    // line of the text has none and may still grow.
    else if (!fence && !line.trim() && offset + line.length < text.length)
      end = offset + line.length + 1;
    offset += line.length + 1;
  }
  return end;
}

// Remove surrounding blank lines without stripping the first line's indentation.
export function displayContent(content: string): string {
  return content.replace(/^(?:[ \t]*\r?\n)+|(?:\r?\n[ \t]*)+$/g, "");
}

export interface MarkdownPainter {
  update(text: string): void;
  reset(): void;
}

// Paint only the part of the markdown that can still change: finished
// blocks append once to .md-done, the open tail repaints in .md-tail, so a
// streamed reply does not rebuild (and de-focus) its earlier DOM per frame.
// Equivalent of chat.html's paintContent; `box` must contain a `.content`
// element. One painter per message element — the state replaces box._md.
export function createMarkdownPainter(box: HTMLElement): MarkdownPainter {
  const state = { done: 0, doneText: "" };
  function update(text: string): void {
    const content = box.querySelector(".content");
    if (!content) return;
    if (!content.firstElementChild) {
      content.innerHTML = '<div class="md-done"></div><div class="md-tail"></div>';
    }
    const [doneEl, tailEl] = content.children;
    const value = displayContent(text);
    if (!value.startsWith(state.doneText)) {
      state.done = 0;
      state.doneText = "";
      doneEl.innerHTML = "";
    }
    const end = mdStableEnd(value);
    if (end > state.done) {
      doneEl.insertAdjacentHTML("beforeend", renderMarkdown(value.slice(state.done, end)));
      state.done = end;
      state.doneText = value.slice(0, end);
    }
    tailEl.innerHTML = renderMarkdown(value.slice(state.done));
  }
  function reset(): void {
    state.done = 0;
    state.doneText = "";
    const content = box.querySelector(".content");
    if (content) content.innerHTML = "";
  }
  return { update, reset };
}
