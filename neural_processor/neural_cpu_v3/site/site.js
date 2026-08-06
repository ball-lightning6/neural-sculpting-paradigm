/* Neural CPU v3 — site behaviour.
   No fetch(), no frameworks. Works over file:// and GitHub Pages.
   Responsibilities:
   1. Bilingual i18n (zh default, en toggle), persisted in localStorage.
   2. Mobile navigation toggle with aria-expanded.
   3. Active navigation state via aria-current.
   4. Copy buttons injected into .codeblock figures. */
(function () {
  "use strict";

  var STORAGE_KEY = "ncv3-lang";

  function dict() {
    var merged = { zh: {}, en: {} };
    var sources = [window.I18N, window.I18N_PAGE];
    for (var s = 0; s < sources.length; s++) {
      var src = sources[s];
      if (!src) continue;
      for (var lang in merged) {
        if (!src[lang]) continue;
        for (var k in src[lang]) merged[lang][k] = src[lang][k];
      }
    }
    return merged;
  }

  function getLang() {
    try {
      var v = window.localStorage.getItem(STORAGE_KEY);
      if (v === "en" || v === "zh") return v;
    } catch (e) { /* file:// or privacy mode — fall through */ }
    return "zh";
  }

  function setStoredLang(lang) {
    try { window.localStorage.setItem(STORAGE_KEY, lang); } catch (e) { /* ignore */ }
  }

  function translate(lang) {
    var d = dict()[lang] || {};
    document.documentElement.lang = (lang === "zh") ? "zh-CN" : "en";

    // text content
    var nodes = document.querySelectorAll("[data-i18n]");
    for (var i = 0; i < nodes.length; i++) {
      var key = nodes[i].getAttribute("data-i18n");
      if (Object.prototype.hasOwnProperty.call(d, key)) {
        nodes[i].textContent = d[key];
      }
    }

    // html content (only for copy we author ourselves)
    var htmlNodes = document.querySelectorAll("[data-i18n-html]");
    for (var j = 0; j < htmlNodes.length; j++) {
      var hkey = htmlNodes[j].getAttribute("data-i18n-html");
      if (Object.prototype.hasOwnProperty.call(d, hkey)) {
        htmlNodes[j].innerHTML = d[hkey];
      }
    }

    // attributes: data-i18n-attr="aria-label:nav.menuAria; content:meta.desc"
    var attrNodes = document.querySelectorAll("[data-i18n-attr]");
    for (var a = 0; a < attrNodes.length; a++) {
      var spec = attrNodes[a].getAttribute("data-i18n-attr").split(";");
      for (var p = 0; p < spec.length; p++) {
        var pair = spec[p].split(":");
        if (pair.length < 2) continue;
        var attr = pair[0].trim();
        var akey = pair.slice(1).join(":").trim();
        if (Object.prototype.hasOwnProperty.call(d, akey)) {
          attrNodes[a].setAttribute(attr, d[akey]);
        }
      }
    }
  }

  function currentLang() {
    return document.documentElement.lang === "en" ? "en" : "zh";
  }

  function setupLangToggle() {
    var btn = document.getElementById("lang-toggle");
    if (!btn) return;
    btn.addEventListener("click", function () {
      var next = currentLang() === "zh" ? "en" : "zh";
      setStoredLang(next);
      translate(next);
    });
  }

  function setupNavToggle() {
    var btn = document.getElementById("nav-toggle");
    var links = document.getElementById("nav-links");
    if (!btn || !links) return;
    btn.addEventListener("click", function () {
      var open = btn.getAttribute("aria-expanded") === "true";
      btn.setAttribute("aria-expanded", String(!open));
      if (open) {
        links.setAttribute("hidden", "");
      } else {
        links.removeAttribute("hidden");
      }
    });
  }

  function markActiveNav() {
    var path = window.location.pathname.replace(/\\/g, "/");
    var file = path.substring(path.lastIndexOf("/") + 1) || "index.html";
    var links = document.querySelectorAll("#nav-links a");
    for (var i = 0; i < links.length; i++) {
      var href = links[i].getAttribute("href") || "";
      var target = href.substring(href.lastIndexOf("/") + 1);
      var inDemosDir = path.indexOf("/demos/") !== -1;
      var match = (target === file) ||
        (target === "demos.html" && inDemosDir);
      if (match) {
        links[i].setAttribute("aria-current", "page");
      } else {
        links[i].removeAttribute("aria-current");
      }
    }
  }

  function setupCopyButtons() {
    var lang = currentLang();
    var d = dict()[lang] || {};
    var blocks = document.querySelectorAll(".codeblock pre");
    for (var i = 0; i < blocks.length; i++) {
      var pre = blocks[i];
      if (pre.parentNode.querySelector(".copy-btn")) continue;
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "copy-btn";
      btn.textContent = d["ui.copy"] || "Copy";
      btn.setAttribute("aria-label", d["ui.copyAria"] || "Copy code block contents");
      btn.setAttribute("title", btn.getAttribute("aria-label"));
      btn.addEventListener("click", function () {
        var code = this.parentNode.querySelector("pre");
        var text = code ? code.textContent : "";
        var self = this;
        function done() {
          var d2 = dict()[currentLang()] || {};
          self.textContent = d2["ui.copied"] || "Copied";
          self.classList.add("copied");
          window.setTimeout(function () {
            var d3 = dict()[currentLang()] || {};
            self.textContent = d3["ui.copy"] || "Copy";
            self.classList.remove("copied");
          }, 1600);
        }
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).then(done, function () { fallbackCopy(text); done(); });
        } else {
          fallbackCopy(text);
          done();
        }
      });
      pre.parentNode.insertBefore(btn, pre);
    }
  }

  function fallbackCopy(text) {
    var ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.left = "-9999px";
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand("copy"); } catch (e) { /* ignore */ }
    document.body.removeChild(ta);
  }

  document.addEventListener("DOMContentLoaded", function () {
    translate(getLang());
    setupLangToggle();
    setupNavToggle();
    markActiveNav();
    setupCopyButtons();

    // copy button labels follow later language switches
    var langBtn = document.getElementById("lang-toggle");
    if (langBtn) {
      langBtn.addEventListener("click", function () {
        var d = dict()[currentLang()] || {};
        var btns = document.querySelectorAll(".copy-btn");
        for (var i = 0; i < btns.length; i++) {
          btns[i].textContent = d["ui.copy"] || "Copy";
          btns[i].setAttribute("aria-label", d["ui.copyAria"] || "Copy code block contents");
          btns[i].setAttribute("title", btns[i].getAttribute("aria-label"));
        }
      });
    }
  });
})();
