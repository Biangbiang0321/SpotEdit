// SpotEdit Grid Mask — a clickable token-grid widget.
// Renders the input draft under an HxW grid; click / drag cells to toggle
// "regenerate" (red) vs "keep". The selection is stored in the node's hidden
// `cells` string widget (row-major 0/1), which the Python node rasterises to a MASK.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function parseCells(str, rows, cols) {
  const g = new Uint8Array(rows * cols);
  if (str && str.length === rows * cols) {
    for (let i = 0; i < str.length; i++) if (str.charCodeAt(i) === 49) g[i] = 1; // '1'
  }
  return g;
}
function cellsToStr(g) {
  let s = "";
  for (let i = 0; i < g.length; i++) s += g[i] ? "1" : "0";
  return s;
}

app.registerExtension({
  name: "SpotEdit.GridMask",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "SpotEditGridMask") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const ret = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      const node = this;

      const cellsW = node.widgets.find((w) => w.name === "cells");
      const colsW = node.widgets.find((w) => w.name === "cols");
      const rowsW = node.widgets.find((w) => w.name === "rows");

      // hide the raw 0/1 string widget but keep it serializable
      if (cellsW) {
        cellsW.computeSize = () => [0, -4];
        cellsW.draw = () => {};
        cellsW.hidden = true;
      }

      node._bg = null;
      node._paintVal = 1;

      const gridWidget = {
        type: "spotedit_grid",
        name: "grid",
        serialize: false,
        draw(ctx, n, w, y) {
          const cols = colsW ? colsW.value : 64;
          const rows = rowsW ? rowsW.value : 64;
          const pad = 8;
          const size = Math.max(32, w - pad * 2);
          const x = pad;
          if (n._bg && n._bg.complete && n._bg.naturalWidth) {
            ctx.drawImage(n._bg, x, y, size, size);
          } else {
            ctx.fillStyle = "#1b1b1b";
            ctx.fillRect(x, y, size, size);
            ctx.fillStyle = "#888";
            ctx.font = "12px sans-serif";
            ctx.fillText("run once to load the draft, then click cells", x + 8, y + 20);
          }
          const cw = size / cols, ch = size / rows;
          const g = parseCells(cellsW ? cellsW.value : "", rows, cols);
          ctx.fillStyle = "rgba(255,45,45,0.45)";
          for (let r = 0; r < rows; r++)
            for (let c = 0; c < cols; c++)
              if (g[r * cols + c]) ctx.fillRect(x + c * cw, y + r * ch, cw, ch);
          if (cols <= 96) {
            ctx.strokeStyle = "rgba(255,255,255,0.12)";
            ctx.lineWidth = 0.5;
            ctx.beginPath();
            for (let c = 0; c <= cols; c++) { ctx.moveTo(x + c * cw, y); ctx.lineTo(x + c * cw, y + size); }
            for (let r = 0; r <= rows; r++) { ctx.moveTo(x, y + r * ch); ctx.lineTo(x + size, y + r * ch); }
            ctx.stroke();
          }
          ctx.strokeStyle = "rgba(255,255,255,0.25)";
          ctx.lineWidth = 1;
          ctx.strokeRect(x, y, size, size);
          n._gridArea = { x, y, size, cw, ch, rows, cols };
        },
        mouse(event, pos, n) {
          const t = event.type;
          if (t !== "pointerdown" && t !== "pointermove") return false;
          if (t === "pointermove" && !(event.buttons & 1)) return false;
          const a = n._gridArea;
          if (!a) return false;
          const lx = pos[0] - a.x, ly = pos[1] - a.y;
          if (lx < 0 || ly < 0 || lx > a.size || ly > a.size) return false;
          const c = Math.min(a.cols - 1, Math.floor(lx / a.cw));
          const r = Math.min(a.rows - 1, Math.floor(ly / a.ch));
          const g = parseCells(cellsW.value, a.rows, a.cols);
          const idx = r * a.cols + c;
          if (t === "pointerdown") n._paintVal = g[idx] ? 0 : 1; // toggle target, drag paints it
          g[idx] = n._paintVal;
          cellsW.value = cellsToStr(g);
          n.setDirtyCanvas(true, true);
          return true;
        },
        computeSize(width) {
          return [width, Math.max(120, width)];
        },
      };
      node.addCustomWidget(gridWidget);

      node.addWidget("button", "clear grid", null, () => {
        if (cellsW) cellsW.value = "";
        node.setDirtyCanvas(true, true);
      });
      node.addWidget("button", "invert grid", null, () => {
        const cols = colsW ? colsW.value : 64;
        const rows = rowsW ? rowsW.value : 64;
        const g = parseCells(cellsW ? cellsW.value : "", rows, cols);
        for (let i = 0; i < g.length; i++) g[i] = g[i] ? 0 : 1;
        cellsW.value = cellsToStr(g);
        node.setDirtyCanvas(true, true);
      });

      node.setSize(node.computeSize());
      return ret;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      if (onExecuted) onExecuted.apply(this, arguments);
      const node = this;
      const cellsW = node.widgets.find((w) => w.name === "cells");
      // adopt python-seeded cells only if the grid is still empty (respect user edits)
      const g = message && message.spotedit_grid && message.spotedit_grid[0];
      if (g && g.seed_cells && cellsW && (!cellsW.value || cellsW.value.length === 0)) {
        cellsW.value = g.seed_cells;
      }
      const im = message && message.images && message.images[0];
      if (im) {
        const url = api.apiURL(
          `/view?filename=${encodeURIComponent(im.filename)}&type=${im.type}&subfolder=${encodeURIComponent(im.subfolder || "")}&rand=${Math.random()}`
        );
        const img = new Image();
        img.onload = () => { node._bg = img; node.setDirtyCanvas(true, true); };
        img.src = url;
      }
    };
  },
});
