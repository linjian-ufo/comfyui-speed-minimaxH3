import { app } from "../../scripts/app.js";

const VALID_SAGE_VALUES = new Set(["auto", "enabled", "disabled"]);

function repairSageWidget(node) {
    const widget = node.widgets?.find((item) => item.name === "sage_attention");
    if (widget && !VALID_SAGE_VALUES.has(widget.value)) {
        widget.value = "auto";
    }
}

app.registerExtension({
    name: "comfyui-speed-minimaxH3.widget-migration",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "MiniMaxH3SpeedCache") {
            return;
        }

        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            repairSageWidget(this);
            return result;
        };

        const originalConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalConfigure?.apply(this, arguments);
            repairSageWidget(this);
            return result;
        };
    },
});
