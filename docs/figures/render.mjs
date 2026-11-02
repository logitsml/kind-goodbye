// Render every Mermaid file (*.mmd) in a folder with pele (https://pele.run) to <name>-light.svg and <name>-dark.svg,
// coloured like GitHub's light and dark themes, for a README's <picture> elements.
//
//     npm install pele && node docs/figures/render.mjs [FIGURES_DIR=docs/figures] [OUT_DIR=docs/images]
//
// pele must be resolvable from this script's folder: run `npm install pele` next to it (or from the repository root
// when the script is copied to docs/figures/render.mjs). Exits 1 and names the file and line on a Mermaid syntax error.
import { render } from 'pele';
import { mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const THEMES = {  // GitHub's Primer colours: page background, text, muted text, lines, surfaces, accent, chart series
  light: { bg: '#ffffff', fg: '#1f2328', muted: '#59636e', line: '#59636e', surface: '#f6f8fa', 'surface-alt': '#eaeef2',
           border: '#d1d9e0', accent: '#0969da', 'series-1': '#0969da', 'series-2': '#bf8700', 'series-3': '#8c959f',
           'series-4': '#1a7f37', 'series-5': '#8250df', 'series-6': '#cf222e' },
  dark: { bg: '#0d1117', fg: '#f0f6fc', muted: '#9198a1', line: '#9198a1', surface: '#151b23', 'surface-alt': '#212830',
          border: '#3d444d', accent: '#4493f8', 'series-1': '#4493f8', 'series-2': '#d29922', 'series-3': '#656c76',
          'series-4': '#3fb950', 'series-5': '#ab7df8', 'series-6': '#f85149' },
};
const FONT = '-apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif';
const dir = process.argv[2] || 'docs/figures', out = process.argv[3] || 'docs/images';
mkdirSync(out, { recursive: true });
let failed = 0;
for (const f of readdirSync(dir).filter((f) => f.endsWith('.mmd')).sort()) {
  const name = f.replace(/\.mmd$/, '');
  try {
    for (const [theme, vars] of Object.entries(THEMES)) {
      const { svg } = render(readFileSync(join(dir, f), 'utf8'), { fontFamily: FONT, idPrefix: `${name}-` });
      // The colours go on the root element itself: an SVG shown through <img> sees no page CSS.
      const css = Object.entries(vars).map(([k, v]) => `--pele-${k}:${v};`).join('') + `--pele-font:${FONT.replaceAll('"', "'")};`;
      writeFileSync(join(out, `${name}-${theme}.svg`), svg.replace('style="', `style="${css}`));
    }
    console.log(name);
  } catch (e) {
    failed++;
    console.error(`${f}: ${e.code || 'error'}${e.line ? ` at line ${e.line}, column ${e.column}` : ''}: ${String(e.message).split('\n')[0]}`);
    if (e.snippet) console.error(e.snippet);
  }
}
process.exit(failed ? 1 : 0);
