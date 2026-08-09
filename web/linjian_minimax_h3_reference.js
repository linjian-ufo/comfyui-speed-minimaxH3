import { app } from "../../scripts/app.js";


const NODE_NAME = "LinjianMiniMaxH3ReferenceToVideo";


app.registerExtension({
  name: "linjian.minimaxH3.referenceToVideoLayout",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const originalCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      originalCreated?.apply(this, arguments);
      // Match the tall built-in Reference-to-Video layout shown in the
      // requested workflow: the multiline prompt occupies the flexible body.
      this.setSize?.([365, 485]);
    };
  },
});
