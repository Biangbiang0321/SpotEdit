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
        // size the backing buffer to the element box (1:1 with CSS px)
        const boxW = Math.max(1, Math.round(cv.clientWidth || cv.width));
        const boxH = Math.max(1, Math.round(cv.clientHeight || cv.height));
        if (cv.width !== boxW) cv.width = boxW;
        if (cv.height !== boxH) cv.height = boxH;
        const ctx = cv.getContext("2d");
        ctx.clearRect(0, 0, boxW, boxH);
        ctx.fillStyle = "#111";
        ctx.fillRect(0, 0, boxW, boxH);
        // SQUARE draw area (image + grid are square) -> no stretch.
        // horizontally centered, TOP-aligned so the image sits right under the
        // buttons instead of floating low when the box is taller than wide.
        const side = Math.min(boxW, boxH);
        const ox = Math.floor((boxW - side) / 2), oy = 0;
        node._area = { ox, oy, side };
        if (node._bg && node._bg.complete && node._bg.naturalWidth) {
          ctx.drawImage(node._bg, ox, oy, side, side);
        } else {
          ctx.fillStyle = "#1b1b1b";
          ctx.fillRect(ox, oy, side, side);
          ctx.fillStyle = "#888";
          ctx.font = "12px sans-serif";
          ctx.fillText("run once to load the draft,", ox + 8, oy + 20);
          ctx.fillText("then click / drag cells", ox + 8, oy + 38);
        }
        const cw = side / cols, ch = side / rows;
        const g = parseCells(cellsW ? cellsW.value : "", rows, cols);
        ctx.fillStyle = "rgba(255,45,45,0.45)";
        for (let r = 0; r < rows; r++)
          for (let c = 0; c < cols; c++)
            if (g[r * cols + c]) ctx.fillRect(ox + c * cw, oy + r * ch, cw, ch);
        if (cols <= 96) {
          ctx.strokeStyle = "rgba(255,255,255,0.12)";
          ctx.lineWidth = 0.5;
          ctx.beginPath();
          for (let c = 0; c <= cols; c++) { ctx.moveTo(ox + c * cw, oy); ctx.lineTo(ox + c * cw, oy + side); }
          for (let r = 0; r <= rows; r++) { ctx.moveTo(ox, oy + r * ch); ctx.lineTo(ox + side, oy + r * ch); }
          ctx.stroke();
        }
        ctx.strokeStyle = "rgba(255,255,255,0.25)";
        ctx.lineWidth = 1;
        ctx.strokeRect(ox + 0.5, oy + 0.5, side - 1, side - 1);
      }
      node._redrawGrid = redraw;

      function cellAt(ev) {
        const rect = cv.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return null;
        const a = node._area || { ox: 0, oy: 0, side: Math.min(cv.width, cv.height) };
        // client -> buffer px (buffer == box after redraw, but scale defensively)
        const sx = cv.width / rect.width, sy = cv.height / rect.height;
        const lx = (ev.clientX - rect.left) * sx - a.ox;
        const ly = (ev.clientY - rect.top) * sy - a.oy;
        if (lx < 0 || ly < 0 || lx > a.side || ly > a.side) return null;
        const c = Math.min(getCols() - 1, Math.max(0, Math.floor(lx / a.side * getCols())));
        const r = Math.min(getRows() - 1, Math.max(0, Math.floor(ly / a.side * getRows())));
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

      // buttons ABOVE the canvas so they are always visible (a tall canvas used
      // to push them off-screen)
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

      const domWidget = node.addDOMWidget("gridcanvas", "spotedit_grid", cv, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => 220,
      });
      // reserve a square canvas (height == width, capped) so the box matches the
      // square image and there's no wasted vertical black; drawing top-aligns anyway
      if (domWidget) domWidget.computeSize = (w) => [w, Math.min(Math.max(200, w), 480)];

      // redraw when the node (and thus the canvas box) is resized
      if (typeof ResizeObserver !== "undefined") {
        const ro = new ResizeObserver(() => redraw());
        try { ro.observe(cv); } catch (e) {}
      }

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
