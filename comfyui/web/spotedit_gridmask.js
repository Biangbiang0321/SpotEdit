// SpotEdit Grid Mask — a clickable token-grid widget.
// Uses a DOM <canvas> (addDOMWidget) so it receives NATIVE pointer events and
// stops them propagating to LiteGraph (otherwise clicks are swallowed and drags
// move the node). Click / drag cells to toggle "regenerate" (red) vs "keep".
// The selection is stored in the node's hidden `cells` string widget (row-major
// 0/1), which the Python node rasterises to a MASK.
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

      // hide the raw 0/1 string widget but keep it serialised with the graph
      if (cellsW) {
        cellsW.computeSize = () => [0, -4];
        cellsW.draw = () => {};
        cellsW.hidden = true;
      }

      node._bg = null;
      node._paintVal = 1;
      node._painting = false;

      // --- DOM canvas ---
      const cv = document.createElement("canvas");
      cv.width = 512;
      cv.height = 512;
      Object.assign(cv.style, {
        width: "100%",
        height: "100%",
        display: "block",
        borderRadius: "4px",
        touchAction: "none",
        cursor: "crosshair",
      });

      const getCols = () => (colsW ? colsW.value : 64);
      const getRows = () => (rowsW ? rowsW.value : 64);

      function redraw() {
        const cols = getCols(), rows = getRows();
        const W = cv.width, H = cv.height;
        const ctx = cv.getContext("2d");
        ctx.clearRect(0, 0, W, H);
        if (node._bg && node._bg.complete && node._bg.naturalWidth) {
          ctx.drawImage(node._bg, 0, 0, W, H);
        } else {
          ctx.fillStyle = "#1b1b1b";
          ctx.fillRect(0, 0, W, H);
          ctx.fillStyle = "#888";
          ctx.font = "13px sans-serif";
          ctx.fillText("run once to load the draft, then click / drag cells", 12, 24);
        }
        const cw = W / cols, ch = H / rows;
        const g = parseCells(cellsW ? cellsW.value : "", rows, cols);
        ctx.fillStyle = "rgba(255,45,45,0.45)";
        for (let r = 0; r < rows; r++)
          for (let c = 0; c < cols; c++)
            if (g[r * cols + c]) ctx.fillRect(c * cw, r * ch, cw, ch);
        if (cols <= 96) {
          ctx.strokeStyle = "rgba(255,255,255,0.12)";
          ctx.lineWidth = 0.5;
          ctx.beginPath();
          for (let c = 0; c <= cols; c++) { ctx.moveTo(c * cw, 0); ctx.lineTo(c * cw, H); }
          for (let r = 0; r <= rows; r++) { ctx.moveTo(0, r * ch); ctx.lineTo(W, r * ch); }
          ctx.stroke();
        }
        ctx.strokeStyle = "rgba(255,255,255,0.25)";
        ctx.lineWidth = 1;
        ctx.strokeRect(0.5, 0.5, W - 1, H - 1);
      }
      node._redrawGrid = redraw;

      function cellAt(ev) {
        const rect = cv.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return null;
        const nx = (ev.clientX - rect.left) / rect.width;
        const ny = (ev.clientY - rect.top) / rect.height;
        if (nx < 0 || ny < 0 || nx > 1 || ny > 1) return null;
        const c = Math.min(getCols() - 1, Math.max(0, Math.floor(nx * getCols())));
        const r = Math.min(getRows() - 1, Math.max(0, Math.floor(ny * getRows())));
        return r * getCols() + c;
      }
      function paint(idx, val) {
        const g = parseCells(cellsW.value, getRows(), getCols());
        g[idx] = val;
        cellsW.value = cellsToStr(g);
        redraw();
      }

      cv.addEventListener("pointerdown", (ev) => {
        ev.stopPropagation();          // keep LiteGraph from starting a node-drag
        ev.preventDefault();
        const idx = cellAt(ev);
        if (idx == null) return;
        const g = parseCells(cellsW.value, getRows(), getCols());
        node._paintVal = g[idx] ? 0 : 1;   // toggle target; drag paints that value
        node._painting = true;
        try { cv.setPointerCapture(ev.pointerId); } catch (e) {}
        paint(idx, node._paintVal);
      });
      cv.addEventListener("pointermove", (ev) => {
        if (!node._painting) return;
        ev.stopPropagation();
        const idx = cellAt(ev);
        if (idx != null) paint(idx, node._paintVal);
      });
      const endPaint = (ev) => {
        if (!node._painting) return;
        node._painting = false;
        try { cv.releasePointerCapture(ev.pointerId); } catch (e) {}
      };
      cv.addEventListener("pointerup", endPaint);
      cv.addEventListener("pointercancel", endPaint);
      cv.addEventListener("pointerleave", (ev) => { if (!(ev.buttons & 1)) endPaint(ev); });
      // swallow context menu / wheel so right-drag & scroll don't hit the graph
      cv.addEventListener("contextmenu", (ev) => ev.preventDefault());

      const domWidget = node.addDOMWidget("gridcanvas", "spotedit_grid", cv, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => 260,
      });
      if (domWidget) domWidget.computeSize = (w) => [w, Math.max(200, w)];

      node.addWidget("button", "reset to judge", null, () => {
        // restore the grid to the judge's auto-suggestion (undo manual edits)
        cellsW.value = node._judgeCells || "";
        redraw();
      });
      node.addWidget("button", "clear grid", null, () => {
        if (cellsW) cellsW.value = "";
        redraw();
      });
      node.addWidget("button", "invert grid", null, () => {
        const g = parseCells(cellsW ? cellsW.value : "", getRows(), getCols());
        for (let i = 0; i < g.length; i++) g[i] = g[i] ? 0 : 1;
        cellsW.value = cellsToStr(g);
        redraw();
      });

      node.setSize(node.computeSize());
      setTimeout(redraw, 0);
      return ret;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      if (onExecuted) onExecuted.apply(this, arguments);
      const node = this;
      const cellsW = node.widgets.find((w) => w.name === "cells");
      const g = message && message.spotedit_grid && message.spotedit_grid[0];
      // remember the judge suggestion so "reset to judge" always has it
      if (g && typeof g.judge_cells === "string") node._judgeCells = g.judge_cells;
      // adopt python-seeded cells only if the grid is still empty (respect user edits)
      if (g && g.seed_cells && cellsW && (!cellsW.value || cellsW.value.length === 0)) {
        cellsW.value = g.seed_cells;
      }
      const im = message && message.images && message.images[0];
      if (im) {
        const url = api.apiURL(
          `/view?filename=${encodeURIComponent(im.filename)}&type=${im.type}&subfolder=${encodeURIComponent(im.subfolder || "")}&rand=${Math.random()}`
        );
        const img = new Image();
        img.onload = () => { node._bg = img; if (node._redrawGrid) node._redrawGrid(); };
        img.src = url;
      } else if (node._redrawGrid) {
        node._redrawGrid();
      }
    };
  },
});
