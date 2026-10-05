"""Render reports/paper.md to a self-contained, print-ready HTML paper
(academic styling; key figures embedded as base64). Open in a browser and
Print -> Save as PDF for a shareable paper.

Run: ./.venv/bin/python src/build_paper_html.py
"""
import base64, re
from pathlib import Path
import markdown

MD = Path("reports/paper.md")
OUT = Path("reports/paper.html")
FIGDIR = Path("reports/figures")


def b64(p):
    return base64.b64encode(Path(p).read_bytes()).decode()


def inline_figures(html: str) -> str:
    """Replace <img src="figures/x.png" alt="cap"> with a self-contained <figure> whose
    caption is the alt text, so figures stay inline in the narrative and the file is portable."""
    def repl(m):
        src, alt = m.group("src"), m.group("alt")
        p = FIGDIR / Path(src).name
        if not p.exists():
            return m.group(0)
        return (f'<figure><img src="data:image/png;base64,{b64(p)}" alt="{alt}"/>'
                f'<figcaption>{alt}</figcaption></figure>')
    # markdown renders images as <img alt="..." src="..." /> (attribute order may vary)
    pat = re.compile(r'<img\s+(?=[^>]*alt="(?P<alt>[^"]*)")(?=[^>]*src="(?P<src>[^"]*)")[^>]*/?>')
    return pat.sub(repl, html)


def main():
    html_body = markdown.markdown(MD.read_text(), extensions=["tables", "toc", "fenced_code"])
    html_body = inline_figures(html_body)
    figs_html = ""  # figures are now embedded inline within the narrative

    css = """
    :root { --ink:#1a1a1a; --muted:#555; --rule:#d0d0d0; --accent:#8b1a1a; }
    * { box-sizing: border-box; }
    body { max-width: 820px; margin: 40px auto; padding: 0 24px;
           font-family: Georgia, 'Times New Roman', serif; color: var(--ink);
           line-height: 1.5; font-size: 16px; }
    h1 { font-size: 1.7em; line-height: 1.25; text-align: center; margin: 0 0 .2em; }
    h2 { font-size: 1.2em; border-bottom: 1px solid var(--rule); padding-bottom: 3px;
         margin-top: 1.8em; }
    h3 { font-size: 1.05em; margin-top: 1.3em; }
    p, li { text-align: justify; }
    strong { color: #000; }
    code { font-family: 'SF Mono', Menlo, monospace; font-size: .85em;
           background: #f4f4f4; padding: 1px 4px; border-radius: 3px; }
    pre { background: #f6f8fa; padding: 12px 14px; border-radius: 6px; overflow-x: auto;
          font-size: .82em; border: 1px solid #eaeaea; }
    pre code { background: none; padding: 0; }
    table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: .9em; }
    th, td { border: 1px solid var(--rule); padding: 6px 10px; text-align: center; }
    th { background: #f2f2f2; }
    td:first-child, th:first-child { text-align: left; }
    figure { margin: 1.5em 0; text-align: center; }
    figure img { max-width: 100%; border: 1px solid #eaeaea; border-radius: 4px; }
    figcaption { font-size: .85em; color: var(--muted); margin-top: 6px; text-align: left; }
    a { color: var(--accent); text-decoration: none; }
    /* the first paragraph after the title (Abstract label lives in body) */
    @media print { body { margin: 0; font-size: 11pt; max-width: 100%; }
                   h2 { page-break-after: avoid; } figure, table, pre { page-break-inside: avoid; } }
    """

    doc = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
           f"<title>Predicting Problematic Internet Use in Youth</title>"
           f"<style>{css}</style></head><body>{html_body}\n{figs_html}</body></html>")
    OUT.write_text(doc)
    n_fig = doc.count("data:image/png;base64")
    print(f"wrote {OUT} ({len(doc)//1024} KB, {n_fig} figures embedded inline)")


if __name__ == "__main__":
    main()
