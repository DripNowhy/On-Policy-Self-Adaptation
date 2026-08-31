(() => {
  const root = document.documentElement;
  const header = document.querySelector(".site-header");
  const progressBar = document.querySelector(".reading-progress span");
  const tocProgress = document.querySelector(".toc-progress span");
  const themeToggle = document.querySelector(".theme-toggle");
  const themeMeta = document.querySelector('meta[name="theme-color"]');
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  const setTheme = (theme, persist = true) => {
    root.dataset.theme = theme;
    themeToggle?.setAttribute(
      "aria-label",
      theme === "dark" ? "Switch to light theme" : "Switch to dark theme",
    );
    themeMeta?.setAttribute("content", theme === "dark" ? "#171714" : "#f8f6f0");

    if (persist) {
      try {
        localStorage.setItem("opsa-theme", theme);
      } catch (_) {}
    }
  };

  setTheme(root.dataset.theme || "light", false);

  themeToggle?.addEventListener("click", () => {
    setTheme(root.dataset.theme === "dark" ? "light" : "dark");
  });

  let scrollFrame = 0;
  const updateScrollState = () => {
    scrollFrame = 0;
    const scrollTop = window.scrollY || document.documentElement.scrollTop;
    const scrollable = document.documentElement.scrollHeight - window.innerHeight;
    const progress = scrollable > 0 ? Math.min(1, Math.max(0, scrollTop / scrollable)) : 0;
    const width = `${(progress * 100).toFixed(3)}%`;

    progressBar?.style.setProperty("width", width);
    tocProgress?.style.setProperty("width", width);
    header?.classList.toggle("is-scrolled", scrollTop > 18);
  };

  const requestScrollUpdate = () => {
    if (!scrollFrame) {
      scrollFrame = window.requestAnimationFrame(updateScrollState);
    }
  };

  updateScrollState();
  window.addEventListener("scroll", requestScrollUpdate, { passive: true });
  window.addEventListener("resize", requestScrollUpdate, { passive: true });

  const revealItems = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window && !reduceMotion.matches) {
    const revealObserver = new IntersectionObserver(
      (entries, observer) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        });
      },
      { rootMargin: "0px 0px -7% 0px", threshold: 0.08 },
    );
    revealItems.forEach((item) => revealObserver.observe(item));
  } else {
    revealItems.forEach((item) => item.classList.add("is-visible"));
  }

  const sections = [...document.querySelectorAll("[data-section]")];
  const tocLinks = [...document.querySelectorAll(".toc a")];

  if ("IntersectionObserver" in window && sections.length) {
    const visibleSections = new Map();
    const sectionObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            visibleSections.set(entry.target.id, entry.boundingClientRect.top);
          } else {
            visibleSections.delete(entry.target.id);
          }
        });

        const active = [...visibleSections.entries()].sort((a, b) => {
          const aDistance = Math.abs(a[1] - 140);
          const bDistance = Math.abs(b[1] - 140);
          return aDistance - bDistance;
        })[0]?.[0];

        if (!active) return;
        tocLinks.forEach((link) => {
          const isActive = link.getAttribute("href") === `#${active}`;
          link.classList.toggle("is-active", isActive);
          if (isActive) link.setAttribute("aria-current", "location");
          else link.removeAttribute("aria-current");
        });
      },
      { rootMargin: "-18% 0px -62% 0px", threshold: [0, 0.15] },
    );
    sections.forEach((section) => sectionObserver.observe(section));
  }

  const dialog = document.querySelector("#image-dialog");
  const dialogImage = dialog?.querySelector(".dialog-inner img");
  const dialogCaption = dialog?.querySelector(".dialog-inner p");
  const zoomTriggers = document.querySelectorAll(".zoom-trigger");

  zoomTriggers.forEach((trigger) => {
    trigger.addEventListener("click", () => {
      const sourceImage = trigger.querySelector("img");
      const source = trigger.dataset.zoom || sourceImage?.src;

      if (!dialog || !dialogImage || !source || typeof dialog.showModal !== "function") {
        if (source) window.open(source, "_blank", "noopener,noreferrer");
        return;
      }

      dialogImage.src = source;
      dialogImage.alt = sourceImage?.alt || "Expanded research figure";
      if (dialogCaption) dialogCaption.textContent = trigger.dataset.caption || "";
      dialog.showModal();
    });
  });

  dialog?.addEventListener("click", (event) => {
    if (event.target !== dialog) return;
    const rect = dialog.getBoundingClientRect();
    const inside =
      event.clientX >= rect.left &&
      event.clientX <= rect.right &&
      event.clientY >= rect.top &&
      event.clientY <= rect.bottom;
    if (!inside) dialog.close();
  });

  dialog?.addEventListener("close", () => {
    if (!dialogImage) return;
    dialogImage.src = "";
    dialogImage.alt = "";
  });

  // Enable reveal styling only after every enhancement above initialized.
  // If this file is blocked, the full article remains visible.
  root.classList.add("reveal-ready");
})();
