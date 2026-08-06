# Notices

`comfyui-speed-minimaxH3` is an independent clean implementation informed by:

- `TE-Speed-MiniMaxH3`: compiled-node interface, sampled signature controls,
  step tracking, cache-device selection, and keyed wrapper behavior.
- [`ComfyUI-MiniMaxH3-Cache`](https://github.com/lihaoyun6/ComfyUI-MiniMaxH3-Cache)
  by lihaoyun6: MiniMax H3 block-loop integration and residual-cache
  approach, released under GPL-3.0.
- [ComfyUI](https://github.com/Comfy-Org/ComfyUI) native `EasyCache` and
  `ModelPatcher` APIs: wrapper lifecycle, timestep preparation patterns,
  model cloning, and patch registration.

No compiled `nodes.pyd` file is included. The project does not copy or replace
ComfyUI's on-disk `comfy/ldm/minimax/model.py` file.
