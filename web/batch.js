import { app } from "../../scripts/app.js";

function pathElement(widget) {
	return widget?.inputEl || widget?.element?.querySelector?.("textarea, input") || null;
}

function fitPathWidget(widget) {
	const el = pathElement(widget);
	if (!el) return;
	el.style.whiteSpace = "pre-wrap";
	el.style.overflowWrap = "anywhere";
	el.style.wordBreak = "break-all";
	el.style.height = "auto";
	el.style.minHeight = "52px";
	el.style.height = `${Math.max(52, el.scrollHeight)}px`;
}

function fitNode(node) {
	for (const widget of node.widgets || []) {
		if (widget.name === "input_folder" || widget.name === "output_folder") fitPathWidget(widget);
	}
	const height = node.computeSize?.()[1];
	if (height) node.size[1] = Math.max(node.size[1], height);
	node.setDirtyCanvas?.(true, true);
}

async function browseInto(node, widgetName) {
	const response = await fetch("/mirror_folder_batch/browse", { method: "POST" });
	const data = await response.json();
	if (!data.path) return;
	const widget = node.widgets?.find((item) => item.name === widgetName);
	if (!widget) return;
	widget.value = data.path;
	widget.callback?.(data.path);
	fitPathWidget(widget);
	fitNode(node);
}

function ensurePanel(node) {
	if (node.etaPanel) return node.etaPanel;
	const panel = document.createElement("div");
	panel.style.cssText = [
		"box-sizing:border-box",
		"width:100%",
		"margin-top:6px",
		"padding:12px 14px 14px",
		"border-radius:10px",
		"background:#1c1c1c",
		"display:flex",
		"flex-direction:column",
		"gap:4px",
		"color:#f2f2f2",
		"font-family:inherit",
	].join(";");

	const left = document.createElement("div");
	left.style.cssText = "font-size:22px;font-weight:600;letter-spacing:-0.02em;line-height:1.15;";
	left.textContent = "—";

	const meta = document.createElement("div");
	meta.style.cssText = "font-size:12px;line-height:1.35;color:#9a9a9a;";
	meta.textContent = "жду второе фото";

	panel.append(left, meta);
	node.addDOMWidget("eta_panel", "ETA", panel, { serialize: false });
	node.etaPanel = { left, meta };
	return node.etaPanel;
}

function showEstimate(node, text) {
	const panel = ensurePanel(node);
	const raw = text || "";
	const photoMatch = raw.match(/это фото\s+([^,]+)/);
	const leftMatch = raw.match(/осталось\s+~?\s*(.+)$/);
	const avgMatch = raw.match(/среднее\s+([^,]+)/);
	const countMatch = raw.match(/(\d+)\s*\/\s*(\d+)/);

	if (photoMatch) {
		panel.left.textContent = photoMatch[1].trim();
		const bits = [];
		if (countMatch) bits.push(`${countMatch[1]} из ${countMatch[2]}`);
		if (leftMatch) bits.push(`осталось ${leftMatch[1].trim()}`);
		if (avgMatch) bits.push(`среднее ${avgMatch[1].trim()}`);
		panel.meta.textContent = bits.join("  ·  ");
	} else if (leftMatch) {
		panel.left.textContent = leftMatch[1].trim();
		const bits = [];
		if (countMatch) bits.push(`${countMatch[1]} из ${countMatch[2]}`);
		if (avgMatch) bits.push(`среднее ${avgMatch[1].trim()}`);
		panel.meta.textContent = bits.join("  ·  ");
	} else {
		panel.left.textContent = "—";
		panel.meta.textContent = raw || "жду второе фото";
	}
	node.setDirtyCanvas(true, true);
}

app.registerExtension({
	name: "mirror.folder.batch",
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData.name !== "MirrorFolderBatch") return;

		const onCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			const result = onCreated?.apply(this, arguments);
			this.addWidget("button", "Обзор входной папки", null, () => browseInto(this, "input_folder"), {
				serialize: false,
			});
			this.addWidget("button", "Обзор папки результата", null, () => browseInto(this, "output_folder"), {
				serialize: false,
			});
			ensurePanel(this);
			this.size = [520, Math.max(this.size?.[1] || 0, 460)];
			requestAnimationFrame(() => fitNode(this));
			return result;
		};

		const onConfigure = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function () {
			const result = onConfigure?.apply(this, arguments);
			requestAnimationFrame(() => fitNode(this));
			return result;
		};

		const onExecuted = nodeType.prototype.onExecuted;
		nodeType.prototype.onExecuted = function (message) {
			onExecuted?.apply(this, arguments);
			showEstimate(this, message?.text?.[0] || "");
		};
	},
});
