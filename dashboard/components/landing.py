"""
Home page landing/marketing sections.

Streamlit pages are server-rendered Python, not React, so "Framer Motion"
doesn't apply literally -- these functions build small, self-contained HTML
fragments (each with its own scoped ``<style>`` and a tiny vanilla-JS reveal
script) meant to be rendered via ``streamlit.components.v1.html``. Every
fragment shares one visual language (``_BASE_STYLE``) so the page reads as a
single cohesive design instead of stitched-together widgets.

Nothing here is dataset-specific or touches app state beyond what is passed
in explicitly -- these are pure HTML builders, easy to reuse or restyle.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Icon system -- plain inline SVG (stroke-based, 24x24 viewBox), used instead
# of emoji everywhere on this page so it reads as a designed product rather
# than a document. No external icon library/network fetch required.
# --------------------------------------------------------------------------- #
_ICON_PATHS = {
    "upload": '<path d="M12 3v11M12 3l-4 4M12 3l4 4"/><path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"/>',
    "filter": (
        '<line x1="4" y1="6" x2="20" y2="6"/><circle cx="9" cy="6" r="1.8"/>'
        '<line x1="4" y1="12" x2="20" y2="12"/><circle cx="15" cy="12" r="1.8"/>'
        '<line x1="4" y1="18" x2="20" y2="18"/><circle cx="7" cy="18" r="1.8"/>'
    ),
    "shield": '<path d="M12 3l7 3v6c0 4.5-3 7.5-7 9-4-1.5-7-4.5-7-9V6l7-3z"/><path d="M9 12l2 2 4-4"/>',
    "cpu": (
        '<rect x="7" y="7" width="10" height="10" rx="2"/>'
        '<path d="M10 3v3M14 3v3M10 18v3M14 18v3M3 10h3M3 14h3M18 10h3M18 14h3"/>'
    ),
    "chart": '<line x1="5" y1="20" x2="5" y2="11"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="19" y1="20" x2="19" y2="15"/>',
    "target": '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r="1" fill="currentColor" stroke="none"/>',
    "dashboard": (
        '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="4.5" rx="1.5"/>'
        '<rect x="14" y="9.5" width="7" height="11.5" rx="1.5"/><rect x="3" y="12" width="7" height="9" rx="1.5"/>'
    ),
    "sparkle": (
        '<path fill="currentColor" stroke="none" '
        'd="M12 2.5l1.7 5.6 5.6 1.7-5.6 1.7-1.7 5.6-1.7-5.6-5.6-1.7 5.6-1.7z"/>'
        '<path fill="currentColor" stroke="none" d="M19 15l.9 2.9 2.9.9-2.9.9-.9 2.9-.9-2.9-2.9-.9 2.9-.9z"/>'
    ),
    "document": (
        '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8l-5-5z"/>'
        '<path d="M14 3v5h5"/><line x1="8" y1="13" x2="16" y2="13"/><line x1="8" y1="17" x2="13" y2="17"/>'
    ),
    "search": '<circle cx="11" cy="11" r="6.5"/><line x1="16" y1="16" x2="21" y2="21"/>',
    "check": '<polyline points="5,13 10,18 19,7"/>',
    "cross": '<line x1="6" y1="6" x2="18" y2="18"/><line x1="18" y1="6" x2="6" y2="18"/>',
}


def _icon(name: str, size: int = 20, stroke_width: float = 1.8) -> str:
    """Render a small inline SVG icon by name (see ``_ICON_PATHS``)."""
    path = _ICON_PATHS[name]
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="currentColor" stroke-width="{stroke_width}" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">{path}</svg>'
    )


# --------------------------------------------------------------------------- #
# Shared visual language
# --------------------------------------------------------------------------- #
# Blue + green + white palette, with a dark-mode variant driven by the
# viewer's OS/browser preference (an embedded iframe cannot reliably read
# Streamlit's own in-app theme toggle across versions, so this is the honest,
# robust signal to key off). Text colors are chosen to clear WCAG AA (>=4.5:1)
# against their background in both modes.
_BASE_STYLE = """
:root{
  --blue:#2563eb; --blue-dark:#1d4ed8; --blue-light:#60a5fa;
  --green:#10b981; --green-dark:#059669;
  --ink:#0f172a; --ink-soft:#3f4c5f; --muted:#5b6b84;
  --bg:#ffffff; --bg-soft:#f8fafc; --card:#ffffff;
  --border:rgba(15,23,42,.09); --shadow:rgba(15,23,42,.10);
  --glass:rgba(255,255,255,.65);
  --section-pad:64px 24px;
}
@media (prefers-color-scheme: dark){
  :root{
    --blue:#60a5fa; --blue-dark:#93c5fd;
    --ink:#f1f5f9; --ink-soft:#c7d2e0; --muted:#93a1b8;
    --bg:#0b1220; --bg-soft:#0f172a; --card:#111a2e;
    --border:rgba(248,250,252,.10); --shadow:rgba(0,0,0,.45);
    --glass:rgba(17,26,46,.6);
  }
}
*{box-sizing:border-box;}
html,body{margin:0; padding:0;}
body{
  background:transparent; color:var(--ink);
  font-family:"Inter","Segoe UI",-apple-system,BlinkMacSystemFont,Roboto,Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased; -moz-osx-font-smoothing:grayscale;
}
.wrap{max-width:1180px; margin:0 auto; padding:0 24px;}
.section{padding:var(--section-pad);}
.eyebrow{
  display:inline-flex; align-items:center; gap:8px; font-size:12.5px; font-weight:700;
  letter-spacing:.05em; text-transform:uppercase; color:var(--blue);
  background:linear-gradient(135deg, rgba(37,99,235,.10), rgba(16,185,129,.10));
  border:1px solid rgba(37,99,235,.20); padding:7px 14px 7px 12px; border-radius:999px;
}
.eyebrow svg{flex:0 0 auto;}
h1,h2,h3{font-weight:800; letter-spacing:-.02em; margin:0; color:var(--ink);}
.gradient-text{
  background:linear-gradient(90deg, var(--blue) 0%, var(--green) 100%);
  -webkit-background-clip:text; background-clip:text; color:transparent;
}
.glass{
  background:var(--glass); backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px);
  border:1px solid var(--border); border-radius:20px;
  box-shadow:0 10px 30px -12px var(--shadow);
}
.card{
  background:var(--card); border:1px solid var(--border); border-radius:16px;
  box-shadow:0 4px 16px -10px var(--shadow);
  transition:transform .3s cubic-bezier(.16,.8,.24,1), box-shadow .3s ease, border-color .3s ease;
}
.card:hover{transform:translateY(-4px); box-shadow:0 16px 32px -14px var(--shadow); border-color:rgba(37,99,235,.30);}
.badge-icon{
  width:48px; height:48px; border-radius:12px; display:flex; align-items:center; justify-content:center;
  color:#fff; background:linear-gradient(135deg, var(--blue), var(--green)); flex:0 0 auto;
  box-shadow:0 6px 16px -6px rgba(37,99,235,.45);
}
.reveal{opacity:0; transform:translateY(22px); transition:opacity .6s cubic-bezier(.16,.8,.24,1), transform .6s cubic-bezier(.16,.8,.24,1);}
body.in-view .reveal{opacity:1; transform:none;}
.section-title{font-size:clamp(24px,3vw,32px); text-align:center; line-height:1.2;}
.section-sub{color:var(--ink-soft); text-align:center; max-width:600px; margin:12px auto 0; font-size:15.5px; line-height:1.65;}
.grid{display:grid; gap:20px;}
@keyframes float{0%,100%{transform:translateY(0px);} 50%{transform:translateY(-14px);}}
@keyframes floatSlow{0%,100%{transform:translateY(0px) rotate(0deg);} 50%{transform:translateY(-8px) rotate(1.5deg);}}
@keyframes pulseGlow{0%,100%{opacity:.5;} 50%{opacity:.9;}}
::-webkit-scrollbar{display:none;}
"""

# Tiny vanilla-JS reveal trigger: uses the PARENT window's IntersectionObserver
# (via ``window.frameElement``) so the animation fires when this component
# actually scrolls into view on the outer Streamlit page -- not the instant it
# mounts. Falls back to an immediate reveal if that cross-frame access is ever
# unavailable, so content is never left invisible.
_REVEAL_SCRIPT = """
<script>
(function(){
  function reveal(){ document.body.classList.add('in-view'); }
  try{
    var IO = window.parent && window.parent.IntersectionObserver;
    var target = window.frameElement;
    if(IO && target){
      var obs = new IO(function(entries){
        if(entries[0].isIntersecting){ reveal(); obs.disconnect(); }
      }, {threshold: 0.1});
      obs.observe(target);
      setTimeout(reveal, 4000);
    } else { reveal(); }
  }catch(e){ reveal(); }
})();
</script>
"""


def _wrap(body: str, extra_style: str = "") -> str:
    """Wrap a section's markup with the shared style + reveal script."""
    return f"<html><head><style>{_BASE_STYLE}{extra_style}</style></head>" \
           f"<body>{body}{_REVEAL_SCRIPT}</body></html>"


def _stagger(index: int, step_ms: int = 80, base_ms: int = 0) -> str:
    """Inline transition-delay for staggered card reveals."""
    return f"transition-delay:{base_ms + index * step_ms}ms"


# --------------------------------------------------------------------------- #
# Hero
# --------------------------------------------------------------------------- #
def render_hero(title: str, subtitle: str) -> str:
    """Hero heading/subtitle + a floating abstract dashboard illustration.

    CTA buttons are intentionally NOT part of this HTML -- they render as real
    ``st.button`` widgets immediately below, so navigation stays native Streamlit.

    The two-column layout collapses to one column below 680px -- calibrated
    against the *actual* rendered width of the embedding iframe (roughly
    750-950px with the sidebar open, not the full browser window), so the
    two-column desktop layout is the one visitors normally see.
    """
    style = """
    .hero{padding:72px 0 20px; display:grid; grid-template-columns:1.05fr .95fr; gap:48px; align-items:center;}
    @media (max-width:680px){.hero{grid-template-columns:1fr; text-align:center; padding-top:48px;}}
    .hero h1{font-size:clamp(28px,4vw,44px); line-height:1.12; margin-top:20px; letter-spacing:-.03em;}
    .hero p{color:var(--ink-soft); font-size:16.5px; line-height:1.7; margin-top:18px; max-width:500px;}
    @media (max-width:680px){.hero p{margin-left:auto; margin-right:auto;}}
    .illustration{position:relative; height:320px;}
    @media (max-width:680px){.illustration{height:280px; margin-top:12px;}}
    .mock{
      position:absolute; inset:0; margin:auto; width:88%; max-width:400px; height:fit-content; padding:20px;
      animation:float 6s ease-in-out infinite;
    }
    .mock-topbar{display:flex; gap:6px; margin-bottom:16px;}
    .dot{width:10px; height:10px; border-radius:50%; background:var(--border);}
    .dot.b1{background:#f87171;} .dot.b2{background:#fbbf24;} .dot.b3{background:#34d399;}
    .kpi-row{display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin-bottom:16px;}
    .kpi-tile{background:var(--bg-soft); border:1px solid var(--border); border-radius:12px; padding:12px 8px;}
    .kpi-tile b{display:block; font-size:16px; color:var(--blue); font-weight:800;}
    .kpi-tile span{font-size:10.5px; color:var(--muted); font-weight:600;}
    .bars{display:flex; align-items:flex-end; gap:8px; height:84px; padding:8px 4px 0;}
    .bars i{flex:1; border-radius:6px 6px 0 0; background:linear-gradient(180deg, var(--green), var(--blue)); opacity:.85;}
    .chip{
      position:absolute; display:inline-flex; align-items:center; gap:6px; padding:9px 14px; border-radius:12px;
      font-size:12px; font-weight:700; color:var(--ink); background:var(--card); border:1px solid var(--border);
      box-shadow:0 10px 24px -10px var(--shadow); animation:floatSlow 7s ease-in-out infinite;
    }
    .chip svg{color:var(--blue); flex:0 0 auto;}
    """
    bars = "".join(f'<i style="height:{h}%"></i>' for h in (38, 62, 48, 82, 56, 70, 44))
    body = f"""
    <div class="wrap hero">
      <div class="reveal">
        <span class="eyebrow">{_icon('sparkle', 13)} AI-Powered Decision Intelligence</span>
        <h1>{title}</h1>
        <p>{subtitle}</p>
      </div>
      <div class="illustration reveal" style="transition-delay:120ms">
        <div class="mock glass">
          <div class="mock-topbar"><span class="dot b1"></span><span class="dot b2"></span><span class="dot b3"></span></div>
          <div class="kpi-row">
            <div class="kpi-tile"><b>94.2%</b><span>Accuracy</span></div>
            <div class="kpi-tile"><b>5</b><span>Models</span></div>
            <div class="kpi-tile"><b>Live</b><span>Pipeline</span></div>
          </div>
          <div class="bars">{bars}</div>
        </div>
        <div class="chip" style="top:4%; right:0%; animation-delay:.4s">{_icon('cpu', 14)} AutoML</div>
        <div class="chip" style="bottom:8%; left:-2%; animation-delay:1.1s">{_icon('chart', 14)} Insights</div>
      </div>
    </div>
    """
    return _wrap(body, style)


# --------------------------------------------------------------------------- #
# Technology stack
# --------------------------------------------------------------------------- #
_TECH_GROUPS: Sequence[Tuple[str, Sequence[Tuple[str, str]]]] = (
    ("Cloud", (("AWS", "Amazon Web Services"), ("GCP", "Google Cloud"), ("Az", "Microsoft Azure"))),
    ("AI & Machine Learning", (
        ("Ge", "Gemini"), ("Py", "Python"), ("Skl", "Scikit-Learn"),
        ("XGB", "XGBoost"), ("LGB", "LightGBM"), ("CB", "CatBoost"),
    )),
    ("Database", (("Pg", "PostgreSQL"), ("Mo", "MongoDB"))),
    ("Frontend", (("Re", "React"), ("Tw", "Tailwind CSS"))),
    ("Backend", (("Fa", "FastAPI"), ("Pd", "Pandas"))),
)


def render_tech_stack() -> str:
    """'Built With Industry-Leading Technologies' -- a monochrome logo-badge showcase.

    Deliberately restrained (no color, no emoji): a neutral outlined badge plus
    the technology name, grouped by category. Every card shares one min-height
    so the grid reads as evenly matched regardless of which row a card lands in
    or how many words its name wraps to.
    """
    style = """
    .tech-group{margin-top:32px;}
    .tech-group:first-of-type{margin-top:0;}
    .tech-group h4{font-size:12.5px; text-transform:uppercase; letter-spacing:.07em; color:var(--muted); font-weight:700; margin:0 0 14px;}
    .tech-grid{display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:14px;}
    .tech-card{padding:18px 14px; text-align:center; cursor:default; min-height:112px; display:flex; flex-direction:column; align-items:center; justify-content:center;}
    .tech-card:hover{transform:translateY(-3px); border-color:rgba(37,99,235,.35); box-shadow:0 14px 28px -16px rgba(37,99,235,.35);}
    .tech-card:hover .mono{border-color:var(--blue); color:var(--blue);}
    .mono{
      width:42px; height:42px; margin:0 auto 12px; border-radius:11px; display:flex; align-items:center;
      justify-content:center; font-weight:700; font-size:12.5px; color:var(--ink-soft);
      background:var(--bg-soft); border:1.5px solid var(--border); transition:border-color .3s ease, color .3s ease;
    }
    .tech-card span{font-size:12.5px; color:var(--ink-soft); font-weight:600; line-height:1.35;}
    .tech-disclaimer{margin-top:32px; font-size:12px; color:var(--muted); text-align:center; line-height:1.6;}
    """
    groups_html = []
    card_index = 0
    for group_name, items in _TECH_GROUPS:
        cards = []
        for mono, full_name in items:
            cards.append(
                f'<div class="tech-card card reveal" style="{_stagger(card_index, 35)}">'
                f'<div class="mono">{mono}</div><span>{full_name}</span></div>'
            )
            card_index += 1
        groups_html.append(f'<div class="tech-group"><h4>{group_name}</h4><div class="tech-grid">{"".join(cards)}</div></div>')

    body = f"""
    <div class="wrap section">
      <div class="reveal" style="text-align:center;">
        <span class="eyebrow">Technology</span>
        <h2 class="section-title" style="margin-top:16px;">Built With Industry-Leading Technologies</h2>
        <p class="section-sub">A modern, production-grade stack spanning cloud, machine learning, data, and application layers.</p>
      </div>
      {"".join(groups_html)}
      <p class="tech-disclaimer">Logos are represented as text badges. These reflect technologies used to build this platform and do not imply official partnership or endorsement.</p>
    </div>
    """
    return _wrap(body, style)


# --------------------------------------------------------------------------- #
# Platform features
# --------------------------------------------------------------------------- #
_FEATURES: Sequence[Tuple[str, str, str]] = (
    ("upload", "Smart Dataset Upload", "Drop in any CSV and let the platform profile it automatically."),
    ("filter", "Automatic Preprocessing", "Missing values, encoding, and scaling handled without configuration."),
    ("shield", "Leakage Detection", "Flags columns that would inflate scores before you ever train a model."),
    ("cpu", "AutoML Training", "Trains and tunes multiple algorithms with zero manual setup."),
    ("chart", "Model Comparison", "Ranks every candidate side by side on the metrics that matter."),
    ("target", "Prediction Engine", "Reuses the saved pipeline to score new data in a single click."),
    ("dashboard", "Executive Dashboard", "Business-friendly KPIs, not just technical metrics."),
    ("sparkle", "AI Decision Support", "Generative AI-assisted insight surfacing -- on the roadmap."),
    ("document", "Business Reports", "Export leaderboards, predictions, and summaries in a click."),
)


def render_features() -> str:
    """Platform capability cards, one per implemented (or roadmap) module."""
    style = """
    .feat-grid{grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); margin-top:36px;}
    .feat-card{padding:26px; display:flex; flex-direction:column;}
    .feat-card h3{font-size:16.5px; margin-top:18px; line-height:1.3; min-height:2.6em;}
    /* Reserve space for two lines even when the text only needs one, so every
       card matches height regardless of description length or which grid row
       it lands in (grid row-stretch alone only equalizes within a shared row). */
    .feat-card p{color:var(--ink-soft); font-size:13.5px; line-height:1.6; margin-top:8px; min-height:3.2em;}
    """
    cards = "".join(
        f'<div class="feat-card card reveal" style="{_stagger(i, 60)}">'
        f'<div class="badge-icon">{_icon(icon, 22)}</div><h3>{name}</h3><p>{desc}</p></div>'
        for i, (icon, name, desc) in enumerate(_FEATURES)
    )
    body = f"""
    <div class="wrap section">
      <div class="reveal" style="text-align:center;">
        <span class="eyebrow">Platform</span>
        <h2 class="section-title" style="margin-top:16px;">Everything You Need, End to End</h2>
        <p class="section-sub">From raw file to boardroom-ready insight -- one connected pipeline.</p>
      </div>
      <div class="grid feat-grid">{cards}</div>
    </div>
    """
    return _wrap(body, style)


# --------------------------------------------------------------------------- #
# How it works
# --------------------------------------------------------------------------- #
_WORKFLOW: Sequence[Tuple[str, str]] = (
    ("upload", "Upload Dataset"),
    ("search", "Smart Analysis"),
    ("cpu", "AutoML Training"),
    ("target", "Prediction"),
    ("dashboard", "Business Dashboard"),
    ("sparkle", "AI Recommendations"),
)


def render_how_it_works() -> str:
    """Horizontal step-by-step workflow with a connecting line that draws in on reveal."""
    style = """
    .flow{position:relative; margin-top:48px; display:flex; justify-content:space-between; gap:6px;}
    .flow-line{position:absolute; top:27px; left:8%; right:8%; height:2px; background:var(--border); border-radius:2px; z-index:0;}
    .flow-line-fill{height:100%; width:0; border-radius:2px; background:linear-gradient(90deg, var(--blue), var(--green)); transition:width 1.6s cubic-bezier(.16,.8,.24,1) .2s;}
    body.in-view .flow-line-fill{width:100%;}
    .step{position:relative; z-index:1; flex:1; text-align:center; display:flex; flex-direction:column; align-items:center;}
    .step .circle{
      width:54px; height:54px; border-radius:50%; display:flex; align-items:center; justify-content:center;
      color:var(--blue); background:var(--card); border:2px solid var(--blue); box-shadow:0 6px 16px -8px var(--shadow);
    }
    .step p{margin-top:12px; font-size:12.5px; font-weight:700; color:var(--ink-soft); line-height:1.4; max-width:110px;}
    @media (max-width:700px){
      .flow{flex-direction:column; align-items:flex-start; gap:26px;}
      .flow-line{display:none;}
      .step{flex-direction:row; text-align:left; gap:14px; max-width:none;}
      .step p{margin-top:0; max-width:none;}
    }
    """
    steps = "".join(
        f'<div class="step reveal" style="{_stagger(i, 120)}"><div class="circle">{_icon(icon, 22)}</div><p>{label}</p></div>'
        for i, (icon, label) in enumerate(_WORKFLOW)
    )
    body = f"""
    <div class="wrap section">
      <div class="reveal" style="text-align:center;">
        <span class="eyebrow">Workflow</span>
        <h2 class="section-title" style="margin-top:16px;">How It Works</h2>
      </div>
      <div class="flow">
        <div class="flow-line"><div class="flow-line-fill"></div></div>
        {steps}
      </div>
    </div>
    """
    return _wrap(body, style)


# --------------------------------------------------------------------------- #
# Statistics (animated counters)
# --------------------------------------------------------------------------- #
def render_stats(stats: Sequence[Tuple[str, object, str, str]]) -> str:
    """Animated stat counters, with support for one non-numeric "stat".

    Args:
        stats: Sequence of (display_label, value, suffix, prefix) tuples. When
            ``value`` is a number it counts up from 0 on reveal (e.g.
            ("Records Processed", 200000, "+", "") -> counts to "200,000+").
            When ``value`` is a string (e.g. "Enterprise") it's shown as-is,
            with the same fade-in treatment but no counting animation.
    """
    style = """
    .stats-grid{grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); margin-top:0;}
    .stat-card{padding:30px 18px; text-align:center; min-height:128px; display:flex; flex-direction:column; align-items:center; justify-content:center;}
    .stat-number{font-size:clamp(26px,3.2vw,36px); font-weight:800; line-height:1.1;}
    .stat-label{margin-top:10px; font-size:13px; color:var(--muted); font-weight:600; letter-spacing:.01em;}
    """
    cards = []
    for i, (label, value, suffix, prefix) in enumerate(stats):
        if isinstance(value, (int, float)):
            number_html = (
                f'<div class="stat-number gradient-text" data-target="{value}" '
                f'data-suffix="{suffix}" data-prefix="{prefix}">{prefix}0{suffix}</div>'
            )
        else:
            number_html = f'<div class="stat-number gradient-text">{value}</div>'
        cards.append(
            f'<div class="stat-card card reveal" style="{_stagger(i, 90)}">'
            f'{number_html}<div class="stat-label">{label}</div></div>'
        )
    count_script = """
    <script>
    (function(){
      function animate(){
        document.querySelectorAll('.stat-number[data-target]').forEach(function(el){
          var target = parseFloat(el.getAttribute('data-target')) || 0;
          var suffix = el.getAttribute('data-suffix') || '';
          var prefix = el.getAttribute('data-prefix') || '';
          var start = performance.now();
          var duration = 1400;
          function tick(now){
            var p = Math.min(1, (now - start) / duration);
            var eased = 1 - Math.pow(1 - p, 3);
            var value = Math.round(target * eased);
            el.textContent = prefix + value.toLocaleString() + suffix;
            if(p < 1) requestAnimationFrame(tick);
          }
          requestAnimationFrame(tick);
        });
      }
      try{
        var IO = window.parent && window.parent.IntersectionObserver;
        var target = window.frameElement;
        if(IO && target){
          var obs = new IO(function(entries){
            if(entries[0].isIntersecting){ document.body.classList.add('in-view'); animate(); obs.disconnect(); }
          }, {threshold: 0.2});
          obs.observe(target);
          setTimeout(function(){ document.body.classList.add('in-view'); animate(); }, 4000);
        } else { document.body.classList.add('in-view'); animate(); }
      }catch(e){ document.body.classList.add('in-view'); animate(); }
    })();
    </script>
    """
    body = f"""
    <div class="wrap" style="padding:56px 24px;">
      <div class="grid stats-grid">{"".join(cards)}</div>
    </div>
    {count_script}
    """
    # Custom wrap (this fragment supplies its own trigger script instead of the shared one).
    return f"<html><head><style>{_BASE_STYLE}{style}</style></head><body>{body}</body></html>"


# --------------------------------------------------------------------------- #
# Comparison: traditional workflow vs. this platform
# --------------------------------------------------------------------------- #
def render_comparison() -> str:
    """'Why Choose Our Platform' -- traditional workflow vs. this platform."""
    style = """
    .cmp-grid{grid-template-columns:1fr 1fr; margin-top:36px;}
    @media (max-width:640px){.cmp-grid{grid-template-columns:1fr;}}
    .cmp-card{padding:30px;}
    .cmp-card h3{font-size:16px; font-weight:700; color:var(--muted); text-transform:uppercase; letter-spacing:.04em;}
    .cmp-card ul{list-style:none; margin:20px 0 0; padding:0;}
    .cmp-card li{display:flex; gap:12px; align-items:center; padding:11px 0; font-size:14.5px; color:var(--ink-soft); font-weight:500; border-top:1px solid var(--border);}
    .cmp-card li:first-of-type{border-top:none;}
    .tag{width:24px; height:24px; border-radius:50%; flex:0 0 auto; display:flex; align-items:center; justify-content:center; color:#fff;}
    .tag.no{background:#ef4444;} .tag.yes{background:var(--green);}
    .tag svg{width:13px; height:13px;}
    .cmp-card.highlight{border-color:rgba(16,185,129,.4); box-shadow:0 16px 36px -18px rgba(16,185,129,.4);}
    .cmp-card.highlight h3{color:var(--green-dark);}
    """
    traditional = ["Manual preprocessing", "Manual model selection", "Technical dashboards", "Requires ML expertise"]
    ours = ["Automatic preprocessing", "AutoML model selection", "Executive dashboard", "AI-powered recommendations"]
    trad_items = "".join(f'<li><span class="tag no">{_icon("cross", 13, 2.4)}</span>{item}</li>' for item in traditional)
    our_items = "".join(f'<li><span class="tag yes">{_icon("check", 13, 2.4)}</span>{item}</li>' for item in ours)
    body = f"""
    <div class="wrap section">
      <div class="reveal" style="text-align:center;">
        <span class="eyebrow">Comparison</span>
        <h2 class="section-title" style="margin-top:16px;">Why Choose Our Platform</h2>
      </div>
      <div class="grid cmp-grid">
        <div class="cmp-card card reveal">
          <h3>Traditional Workflow</h3>
          <ul>{trad_items}</ul>
        </div>
        <div class="cmp-card card highlight reveal" style="transition-delay:110ms">
          <h3>Our Platform</h3>
          <ul>{our_items}</ul>
        </div>
      </div>
    </div>
    """
    return _wrap(body, style)


# --------------------------------------------------------------------------- #
# CTA banner
# --------------------------------------------------------------------------- #
def render_cta_banner(heading: str) -> str:
    """Large closing call-to-action banner (buttons render as real st.button below it)."""
    style = """
    .cta{
      margin:8px 24px 0; padding:60px 32px; border-radius:26px; text-align:center;
      background:linear-gradient(135deg, var(--blue) 0%, var(--green) 100%);
      position:relative; overflow:hidden;
    }
    .cta h2{color:#fff; font-size:clamp(23px,3.2vw,32px); max-width:640px; margin:0 auto; line-height:1.25;}
    .cta .glow{position:absolute; width:240px; height:240px; border-radius:50%; background:rgba(255,255,255,.16); filter:blur(20px); animation:pulseGlow 4s ease-in-out infinite;}
    """
    body = f"""
    <div class="cta reveal">
      <div class="glow" style="top:-70px; left:-50px;"></div>
      <div class="glow" style="bottom:-80px; right:-30px; animation-delay:1.5s;"></div>
      <h2 style="position:relative;">{heading}</h2>
    </div>
    """
    return _wrap(body, style)


# --------------------------------------------------------------------------- #
# Footer
# --------------------------------------------------------------------------- #
def render_footer(app_title: str) -> str:
    """Professional footer with informational link columns."""
    style = """
    .footer{margin-top:16px; padding:44px 24px 32px; border-top:1px solid var(--border);}
    .footer-grid{display:grid; grid-template-columns:1.5fr 1fr 1fr 1fr; gap:28px;}
    @media (max-width:640px){.footer-grid{grid-template-columns:1fr 1fr;}}
    .footer h5{font-size:12px; text-transform:uppercase; letter-spacing:.07em; color:var(--muted); font-weight:700; margin:0 0 14px;}
    .footer a{display:block; color:var(--ink-soft); text-decoration:none; font-size:13.5px; font-weight:500; padding:6px 0; transition:color .2s ease;}
    .footer a:hover{color:var(--blue);}
    .footer-brand p{color:var(--muted); font-size:13px; line-height:1.65; max-width:280px; margin:0;}
    .footer-bottom{display:flex; justify-content:space-between; align-items:center; margin-top:32px; padding-top:22px; border-top:1px solid var(--border); flex-wrap:wrap; gap:12px;}
    .footer-bottom span{font-size:12.5px; color:var(--muted);}
    .footer-legal{display:flex; gap:20px;}
    .footer-legal a{padding:0; font-size:12.5px; font-weight:500; color:var(--muted);}
    .footer-legal a:hover{color:var(--blue);}
    """
    body = f"""
    <div class="wrap footer reveal">
      <div class="footer-grid">
        <div class="footer-brand">
          <h5>{app_title}</h5>
          <p>An AI-powered decision intelligence platform that turns raw business data into executive-ready insight.</p>
        </div>
        <div><h5>About</h5><a href="#">Our Mission</a><a href="#">How It Works</a></div>
        <div><h5>Features</h5><a href="#">AutoML Engine</a><a href="#">Executive Dashboard</a></div>
        <div><h5>Technology</h5><a href="#">Tech Stack</a><a href="#">Contact</a></div>
      </div>
      <div class="footer-bottom">
        <span>&copy; 2026 {app_title}. Built for a national datathon.</span>
        <div class="footer-legal">
          <a href="#">GitHub</a>
          <a href="#">LinkedIn</a>
          <a href="#">Privacy Policy</a>
        </div>
      </div>
    </div>
    """
    return _wrap(body, style)
