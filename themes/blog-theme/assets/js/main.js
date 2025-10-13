document.addEventListener("DOMContentLoaded", () => {
  const tocSidebar = document.getElementById("toc-sidebar");
  const postContent = document.getElementById("post-content");

  if (!tocSidebar || !postContent) {
    return;
  }

  const alignToc = () => {
    const target = postContent.querySelector(
      "p:not(.context-figure__note):not([data-ignore-toc])"
    );
    const fallback =
      postContent.querySelector("h2") || postContent.firstElementChild;
    const anchor = target || fallback;
    if (!anchor) return;

    const articleRect = postContent.getBoundingClientRect();
    const targetRect = anchor.getBoundingClientRect();
    let delta = targetRect.top - articleRect.top;

    const cs = window.getComputedStyle(anchor);
    const mt = parseFloat(cs.marginTop || "0") || 0;
    delta = Math.max(0, delta - mt);

    tocSidebar.style.marginTop = `${delta}px`;
  };

  alignToc();
  window.addEventListener("resize", alignToc);

  // Ensure page title appears in TOC (insert at top if missing); still hide subtitle
  const pageTitle = tocSidebar.getAttribute("data-title");
  const pageSubtitle = tocSidebar.getAttribute("data-subtitle");
  const tocRoot = tocSidebar.querySelector("#TableOfContents > ul");
  if (tocRoot && pageTitle) {
    const hasTitle = Array.from(tocRoot.querySelectorAll("a")).some(
      (a) => (a.textContent || "").trim() === pageTitle
    );
    if (!hasTitle) {
      const li = document.createElement("li");
      const a = document.createElement("a");
      a.href = "#post-title";
      a.textContent = pageTitle;
      li.appendChild(a);
      tocRoot.insertBefore(li, tocRoot.firstChild);
    }
  }

  // Hide subtitle entry from TOC if it exists
  if (pageSubtitle) {
    const subtitleLink = Array.from(
      tocSidebar.querySelectorAll("#TableOfContents a")
    ).find((a) => (a.textContent || "").trim() === pageSubtitle);
    if (subtitleLink) {
      const li = subtitleLink.closest("li");
      if (li) li.style.display = "none";
    }
  }
});

// Add copy-to-clipboard buttons to code blocks
document.addEventListener("DOMContentLoaded", () => {
  const codeBlocks = document.querySelectorAll("pre > code");
  if (!codeBlocks || codeBlocks.length === 0) return;

  const clipboardSvg =
    '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" focusable="false"><path d="M9 4h6a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H9a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2zm1-2h4a1 1 0 0 1 1 1v1H9V3a1 1 0 0 1 1-1z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  const checkSvg =
    '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" focusable="false"><path d="M5 13l4 4L19 7" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';

  const setCopiedState = (button) => {
    const originalAria = button.getAttribute("aria-label") || "Copy code";
    button.innerHTML = checkSvg;
    button.classList.add("copied");
    button.setAttribute("aria-label", "Copied");
    setTimeout(() => {
      button.classList.remove("copied");
      button.innerHTML = clipboardSvg;
      button.setAttribute("aria-label", originalAria);
    }, 1500);
  };

  codeBlocks.forEach((code) => {
    const pre = code.parentElement;
    if (!pre || pre.querySelector(".copy-btn")) return;

    const button = document.createElement("button");
    button.type = "button";
    button.className = "copy-btn";
    button.setAttribute("aria-label", "Copy code");
    button.innerHTML = clipboardSvg;

    pre.appendChild(button);

    button.addEventListener("click", async () => {
      const text = code.innerText || code.textContent || "";
      try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          await navigator.clipboard.writeText(text);
        } else {
          throw new Error("Clipboard API not available");
        }
        setCopiedState(button);
      } catch (_) {
        // Fallback for older browsers
        const textarea = document.createElement("textarea");
        textarea.value = text;
        textarea.style.position = "fixed";
        textarea.style.top = "-1000px";
        textarea.style.left = "-1000px";
        document.body.appendChild(textarea);
        textarea.focus();
        textarea.select();
        try {
          document.execCommand("copy");
          setCopiedState(button);
        } finally {
          document.body.removeChild(textarea);
        }
      }
    });
  });
});
