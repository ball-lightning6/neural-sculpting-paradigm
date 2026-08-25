(() => {
  const body = document.body;
  const navToggle = document.querySelector("[data-nav-toggle]");
  const sidebar = document.getElementById("sidebar");

  const setNavOpen = (open) => {
    body.classList.toggle("nav-open", open);
    navToggle?.setAttribute("aria-expanded", String(open));
  };

  navToggle?.addEventListener("click", () => {
    setNavOpen(!body.classList.contains("nav-open"));
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") setNavOpen(false);
  });

  document.addEventListener("click", (event) => {
    if (!body.classList.contains("nav-open")) return;
    if (sidebar?.contains(event.target) || navToggle?.contains(event.target)) return;
    setNavOpen(false);
  });

  sidebar?.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", () => setNavOpen(false));
  });

  const progress = document.querySelector(".reading-progress span");
  const updateProgress = () => {
    if (!progress) return;
    const root = document.documentElement;
    const scrollable = Math.max(1, root.scrollHeight - root.clientHeight);
    const ratio = Math.min(1, Math.max(0, root.scrollTop / scrollable));
    progress.style.width = `${ratio * 100}%`;
  };
  updateProgress();
  document.addEventListener("scroll", updateProgress, { passive: true });
  window.addEventListener("resize", updateProgress);

  const tocLinks = [...document.querySelectorAll('.toc a[href^="#"]')];
  const headings = tocLinks
    .map((link) => document.getElementById(decodeURIComponent(link.hash.slice(1))))
    .filter(Boolean);

  if (headings.length && "IntersectionObserver" in window) {
    const linkById = new Map(tocLinks.map((link) => [decodeURIComponent(link.hash.slice(1)), link]));
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (!visible.length) return;
        tocLinks.forEach((link) => link.classList.remove("active"));
        linkById.get(visible[0].target.id)?.classList.add("active");
      },
      { rootMargin: "-72px 0px -72% 0px", threshold: [0, 1] },
    );
    headings.forEach((heading) => observer.observe(heading));
  }

  const search = document.querySelector("[data-experiment-search]");
  const cards = [...document.querySelectorAll("[data-experiment-card]")];
  const count = document.querySelector("[data-search-count]");
  const empty = document.querySelector("[data-empty-search]");

  const filterExperiments = () => {
    if (!search) return;
    const query = search.value.trim().toLocaleLowerCase("zh-CN");
    let visible = 0;
    cards.forEach((card) => {
      const matches = !query || card.dataset.searchText.includes(query);
      card.hidden = !matches;
      if (matches) visible += 1;
    });
    if (count) count.textContent = `${visible} / ${cards.length}`;
    if (empty) empty.hidden = visible !== 0;
  };

  search?.addEventListener("input", filterExperiments);
})();
