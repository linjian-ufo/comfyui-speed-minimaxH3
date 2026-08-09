import { app } from "../../scripts/app.js";

const BUILDER_NODE = "LinjianMiniMaxH3ImageToVideo";
const SUBGRAPH_NAME = "linjian Image to Video (MiniMax H3)";
const DEFAULT_PROMPT = "根据图片顺序1-9，每图片中的分镜头提示词生成视频和声音";
const DEFAULT_SEED = 28250570182816;
const IMAGE_TOOLTIPS_ZH = Object.freeze({
    first_frame: "作用：可选的首帧图片，用于约束视频开头的主体、构图和风格。默认不连接；没有数值范围或步长。连接后会按 width、height 处理；图片与目标比例差异过大时可能裁切或缩放。",
    last_frame: "作用：可选的尾帧图片，用于约束视频结尾画面和首尾过渡。默认不连接；没有数值范围或步长。只接首帧是普通图生视频，同时接首尾帧会增加约束，跨度过大可能降低运动自然度。",
    prompt: `作用：描述视频内容、镜头运动、动作、环境和声音。默认值：${DEFAULT_PROMPT}。没有数值范围或步长；描述越明确越容易稳定复现，互相冲突或过长的提示词可能降低一致性。`,
    width: "作用：设置输出视频宽度。默认值：1344；建议下限 32、上限 16384、步长 32。数值越大细节潜力越高，但显存占用和生成时间会明显增加；建议保持 32 的倍数。外接宽度连线时以连线值为准。",
    height: "作用：设置输出视频高度。默认值：768；建议下限 32、上限 16384、步长 32。数值越大细节潜力越高，但显存占用和生成时间会明显增加；建议保持 32 的倍数。外接高度连线时以连线值为准。",
    value_1: "作用：设置视频时长并按 24 fps 换算为符合 MiniMax H3 结构的帧数。默认值：30 秒；建议首次测试 5 秒，常用范围约 5～15 秒，调节步长 0.1 秒。时长越长，帧数、显存和生成时间近似成倍增加；30 秒会生成约 736 帧，明显慢于 5 秒的约 124 帧。",
    steps: "作用：控制 BasicScheduler 的扩散采样步数。默认值：20；下限 1；上限 10000；步长 1。步数越高通常有更多收敛机会，但生成时间近似增加；步数过低可能出现细节不足、运动不稳定。MiniMax H3 建议先用 20，再按画质和速度小幅调整。",
    noise_seed: `作用：控制初始噪声，以便复现或变化结果。默认值：${DEFAULT_SEED}；范围 0～18446744073709551615；步长 1。相同模型、参数、输入和种子通常便于对比；更换种子会改变构图、动作和细节，不代表画质必然提高。`,
    unet_name: "作用：接收已经经过 MiniMax H3 Speed Cache (Safe) 的 MODEL，并同时送入内部调度器和引导器。默认无连接，没有数值范围或步长。不要直接连接原始 UNETLoader，也不要串联其他 block_loop 缓存；正确连接为 UNETLoader → MiniMax H3 加速缓存 → 本接口。",
    cache_device: "作用：覆盖外接 MiniMax H3 加速节点本次运行的缓存位置。默认 auto；可选 auto、gpu、cpu，没有数值范围或步长。auto 根据可用显存/内存自动选择，推荐；gpu 通常最快但更占显存；cpu 更省显存，但缓存复用时传回显卡可能使长视频变慢。",
    verbose: "作用：覆盖外接 MiniMax H3 加速节点的详细日志开关。默认关闭（false）；可选开启或关闭，没有数值范围或步长。开启后控制台显示每步 RUN/SKIP、实际缓存设备、跳步数量及估算加速比，适合首次测速和排错；日常使用可关闭。",
    clip_name: "作用：选择 MiniMax H3 的 QwenVL 文本/视觉编码器。默认使用 qwen3vl_32b MiniMax H3 INT8 ConvRot 文件；没有数值范围或步长。模型必须与 MiniMax H3 的 minimax 类型兼容；更换文件会触发重新加载，首次编码仍可能较慢。",
    vae_name: "作用：选择 MiniMax H3 视频 VAE，用于把潜空间解码为视频帧。默认 minimax_h3_video_vae_fp16.safetensors；没有数值范围或步长。错误或不匹配的 VAE 会导致解码失败、颜色异常或额外内存占用。",
    vae_name_1: "作用：选择 MiniMax H3 音频 VAE，用于把音频潜空间解码为声音。默认 minimax_h3_audio_vae_fp32.safetensors；没有数值范围或步长。错误或不匹配的音频 VAE 会导致无声、解码失败或音频异常。",
});
let imageNodeDefUpdateChain = Promise.resolve();

const input = (name, type, link, widget = false, optional = false) => ({
    name,
    type,
    link,
    ...(widget ? { widget: { name } } : {}),
    ...(optional ? { shape: 7 } : {}),
});

const output = (name, type, links) => ({ name, type, links });

function node(id, type, pos, size, order, inputs, outputs, widgetsValues = [], title) {
    return {
        id,
        type,
        pos,
        size,
        flags: {},
        order,
        mode: 0,
        inputs,
        outputs,
        properties: {
            cnr_id: "comfy-core",
            ver: "0.30.0",
            "Node name for S&R": type,
        },
        widgets_values: widgetsValues,
        ...(title ? { title } : {}),
    };
}

function createSubgraphDefinition() {
    const slot = (name, type, linkIds, pos, label, tooltip) => ({
        id: crypto.randomUUID(),
        name,
        type,
        linkIds,
        pos,
        ...(label ? { label } : {}),
        ...(tooltip ? { tooltip } : {}),
    });

    const nodes = [
        node(
            11,
            "VAELoader",
            [-2020, 4970],
            [640, 70],
            4,
            [input("vae_name", "COMBO", 223, true)],
            [output("VAE", "VAE", [8, 190])],
            ["minimax-H3\\minimax_h3_video_vae_fp16.safetensors"],
        ),
        node(
            24,
            "VAELoader",
            [-2020, 5100],
            [650, 70],
            10,
            [input("vae_name", "COMBO", 224, true)],
            [output("VAE", "VAE", [23])],
            ["minimax-H3\\minimax_h3_audio_vae_fp32.safetensors"],
        ),
        node(
            23,
            "VAEDecodeAudio",
            [-50, 4880],
            [230, 60],
            9,
            [input("samples", "LATENT", 226), input("vae", "VAE", 23)],
            [output("AUDIO", "AUDIO", [166])],
        ),
        node(
            10,
            "VAEDecode",
            [-50, 4760],
            [230, 60],
            3,
            [input("samples", "LATENT", 225), input("vae", "VAE", 8)],
            [output("IMAGE", "IMAGE", [167])],
        ),
        node(
            17,
            "KSamplerSelect",
            [-790, 4910],
            [370, 70],
            0,
            [input("sampler_name", "COMBO", null, true)],
            [output("SAMPLER", "SAMPLER", [16])],
            ["res_multistep"],
        ),
        node(
            9,
            "BasicScheduler",
            [-790, 5030],
            [370, 130],
            2,
            [
                input("model", "MODEL", 5),
                input("scheduler", "COMBO", null, true),
                input("steps", "INT", 232, true),
                input("denoise", "FLOAT", null, true),
            ],
            [output("SIGMAS", "SIGMAS", [18])],
            ["simple", 20, 1],
        ),
        node(
            113,
            "MiniMaxH3CacheRuntimeOptions",
            [-2020, 4540],
            [640, 150],
            16,
            [
                input("model", "MODEL", 229),
                input("cache_device", "COMBO", 230, true),
                input("verbose", "BOOLEAN", 231, true),
            ],
            [output("model", "MODEL", [5, 193])],
            ["auto", false],
            "子图缓存运行选项",
        ),
        node(
            14,
            "SamplerCustomAdvanced",
            [-360, 4820],
            [230, 140],
            6,
            [
                input("noise", "NOISE", 40),
                input("guider", "GUIDER", 12),
                input("sampler", "SAMPLER", 16),
                input("sigmas", "SIGMAS", 18),
                input("latent_image", "LATENT", 188),
            ],
            [output("output", "LATENT", [225, 226]), output("denoised_output", "LATENT", null)],
        ),
        node(
            16,
            "BasicGuider",
            [-790, 4800],
            [360, 60],
            8,
            [input("model", "MODEL", 193), input("conditioning", "CONDITIONING", 187)],
            [output("GUIDER", "GUIDER", [12])],
        ),
        node(
            13,
            "CLIPLoader",
            [-2020, 4780],
            [640, 120],
            5,
            [
                input("clip_name", "COMBO", 227, true),
                input("type", "COMBO", null, true),
                input("device", "COMBO", null, true, true),
            ],
            [output("CLIP", "CLIP", [228])],
            ["minimax-H3\\qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "minimax", "default"],
        ),
        node(
            112,
            "MiniMaxH3TextEncoderCache",
            [-1290, 5260],
            [410, 180],
            15,
            [
                input("clip", "CLIP", 228),
                input("cache_enabled", "BOOLEAN", null, true),
                input("max_cache_entries", "INT", null, true),
                input("verbose", "BOOLEAN", null, true, true),
                input("sage_attention", "COMBO", null, true, true),
            ],
            [output("clip", "CLIP", [189])],
            [true, 2, false, "auto"],
            "QwenVL SageAttention 2.1.1",
        ),
        node(
            15,
            "RandomNoise",
            [-790, 4660],
            [360, 90],
            7,
            [input("noise_seed", "INT", 207, true)],
            [output("NOISE", "NOISE", [40])],
            [DEFAULT_SEED, "randomize"],
        ),
        node(
            91,
            "CreateVideo",
            [260, 4790],
            [270, 110],
            11,
            [
                input("images", "IMAGE", 167),
                input("audio", "AUDIO", 166, false, true),
                input("fps", "FLOAT", null, true),
                input("bit_depth", "INT", null, true, true),
            ],
            [output("VIDEO", "VIDEO", [168])],
            [24, 8],
        ),
        node(
            104,
            "MiniMaxH3ImageToVideo",
            [-1290, 4650],
            [410, 510],
            12,
            [
                input("clip", "CLIP", 189),
                input("vae", "VAE", 190),
                input("first_frame", "IMAGE", 195, false, true),
                input("last_frame", "IMAGE", 196, false, true),
                input("prompt", "STRING", 197, true),
                input("width", "INT", 200, true),
                input("height", "INT", 201, true),
                input("length", "INT", 199, true),
            ],
            [output("positive", "CONDITIONING", [187]), output("LATENT", "LATENT", [188])],
            [
                DEFAULT_PROMPT,
                1344,
                768,
                736,
            ],
        ),
        node(
            107,
            "ComfyMathExpression",
            [-1710, 5300],
            [360, 160],
            13,
            [
                { ...input("values.a", "FLOAT,INT,BOOLEAN", 205), label: "a" },
                { ...input("values.b", "FLOAT,INT,BOOLEAN", null, false, true), label: "b" },
                input("expression", "STRING", null, true),
            ],
            [
                output("FLOAT", "FLOAT", null),
                output("INT", "INT", [199]),
                output("BOOL", "BOOLEAN", null),
            ],
            ["max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17"],
        ),
        node(
            111,
            "PrimitiveFloat",
            [-2020, 5300],
            [270, 70],
            14,
            [input("value", "FLOAT", 206, true)],
            [output("FLOAT", "FLOAT", [205])],
            [30],
            "Float (duration)",
        ),
    ];

    return {
        id: crypto.randomUUID(),
        version: 1,
        state: { lastGroupId: 5, lastNodeId: 113, lastLinkId: 232, lastRerouteId: 0 },
        revision: 0,
        config: {},
        name: SUBGRAPH_NAME,
        description: "Subgraph node for linjian Image to Video (MiniMax H3); connect the accelerated MODEL to unet_name.",
        inputNode: { id: -10, bounding: [-2560, 4680, 128, 328] },
        outputNode: { id: -20, bounding: [670, 4780, 128, 68] },
        inputs: [
            slot("first_frame", "IMAGE", [195], [-2456, 4704], null, IMAGE_TOOLTIPS_ZH.first_frame),
            slot("last_frame", "IMAGE", [196], [-2456, 4724], null, IMAGE_TOOLTIPS_ZH.last_frame),
            slot("prompt", "STRING", [197], [-2456, 4744], null, IMAGE_TOOLTIPS_ZH.prompt),
            slot("width", "INT", [200], [-2456, 4764], null, IMAGE_TOOLTIPS_ZH.width),
            slot("height", "INT", [201], [-2456, 4784], null, IMAGE_TOOLTIPS_ZH.height),
            slot("value_1", "FLOAT", [206], [-2456, 4804], "duration", IMAGE_TOOLTIPS_ZH.value_1),
            slot("steps", "INT", [232], [-2456, 4824], null, IMAGE_TOOLTIPS_ZH.steps),
            slot("noise_seed", "INT", [207], [-2456, 4844], null, IMAGE_TOOLTIPS_ZH.noise_seed),
            slot("unet_name", "MODEL", [229], [-2456, 4864], null, IMAGE_TOOLTIPS_ZH.unet_name),
            slot("cache_device", "COMBO", [230], [-2456, 4884], null, IMAGE_TOOLTIPS_ZH.cache_device),
            slot("verbose", "BOOLEAN", [231], [-2456, 4904], null, IMAGE_TOOLTIPS_ZH.verbose),
            slot("clip_name", "COMBO", [227], [-2456, 4924], null, IMAGE_TOOLTIPS_ZH.clip_name),
            slot("vae_name", "COMBO", [223], [-2456, 4944], null, IMAGE_TOOLTIPS_ZH.vae_name),
            slot("vae_name_1", "COMBO", [224], [-2456, 4964], "audio_vae", IMAGE_TOOLTIPS_ZH.vae_name_1),
        ],
        outputs: [
            {
                id: crypto.randomUUID(),
                name: "VIDEO",
                type: "VIDEO",
                linkIds: [168],
                localized_name: "VIDEO",
                pos: [694, 4804],
            },
        ],
        widgets: [],
        nodes,
        groups: [
            { id: 1, title: "Models", bounding: [-2050, 4540, 700, 670], color: "#3f789e", flags: {} },
            { id: 2, title: "Sampling", bounding: [-810, 4540, 690, 670], color: "#3f789e", flags: {} },
            { id: 3, title: "Conditioning", bounding: [-1320, 4540, 480, 670], color: "#3f789e", flags: {} },
            { id: 4, title: "Decoding and create video", bounding: [-90, 4540, 670, 670], color: "#3f789e", flags: {} },
            { id: 5, title: "QwenVL / SageAttention 2.1.1", bounding: [-1320, 5220, 480, 260], color: "#3f789e", flags: {} },
        ],
        links: [
            { id: 23, origin_id: 24, origin_slot: 0, target_id: 23, target_slot: 1, type: "VAE" },
            { id: 8, origin_id: 11, origin_slot: 0, target_id: 10, target_slot: 1, type: "VAE" },
            { id: 5, origin_id: 113, origin_slot: 0, target_id: 9, target_slot: 0, type: "MODEL" },
            { id: 40, origin_id: 15, origin_slot: 0, target_id: 14, target_slot: 0, type: "NOISE" },
            { id: 12, origin_id: 16, origin_slot: 0, target_id: 14, target_slot: 1, type: "GUIDER" },
            { id: 16, origin_id: 17, origin_slot: 0, target_id: 14, target_slot: 2, type: "SAMPLER" },
            { id: 18, origin_id: 9, origin_slot: 0, target_id: 14, target_slot: 3, type: "SIGMAS" },
            { id: 188, origin_id: 104, origin_slot: 1, target_id: 14, target_slot: 4, type: "LATENT" },
            { id: 193, origin_id: 113, origin_slot: 0, target_id: 16, target_slot: 0, type: "MODEL" },
            { id: 187, origin_id: 104, origin_slot: 0, target_id: 16, target_slot: 1, type: "CONDITIONING" },
            { id: 167, origin_id: 10, origin_slot: 0, target_id: 91, target_slot: 0, type: "IMAGE" },
            { id: 166, origin_id: 23, origin_slot: 0, target_id: 91, target_slot: 1, type: "AUDIO" },
            { id: 228, origin_id: 13, origin_slot: 0, target_id: 112, target_slot: 0, type: "CLIP" },
            { id: 189, origin_id: 112, origin_slot: 0, target_id: 104, target_slot: 0, type: "CLIP" },
            { id: 190, origin_id: 11, origin_slot: 0, target_id: 104, target_slot: 1, type: "VAE" },
            { id: 168, origin_id: 91, origin_slot: 0, target_id: -20, target_slot: 0, type: "VIDEO" },
            { id: 195, origin_id: -10, origin_slot: 0, target_id: 104, target_slot: 2, type: "IMAGE" },
            { id: 196, origin_id: -10, origin_slot: 1, target_id: 104, target_slot: 3, type: "IMAGE" },
            { id: 197, origin_id: -10, origin_slot: 2, target_id: 104, target_slot: 4, type: "STRING" },
            { id: 199, origin_id: 107, origin_slot: 1, target_id: 104, target_slot: 7, type: "INT" },
            { id: 200, origin_id: -10, origin_slot: 3, target_id: 104, target_slot: 5, type: "INT" },
            { id: 201, origin_id: -10, origin_slot: 4, target_id: 104, target_slot: 6, type: "INT" },
            { id: 205, origin_id: 111, origin_slot: 0, target_id: 107, target_slot: 0, type: "FLOAT" },
            { id: 206, origin_id: -10, origin_slot: 5, target_id: 111, target_slot: 0, type: "FLOAT" },
            { id: 207, origin_id: -10, origin_slot: 7, target_id: 15, target_slot: 0, type: "INT" },
            { id: 223, origin_id: -10, origin_slot: 12, target_id: 11, target_slot: 0, type: "COMBO" },
            { id: 224, origin_id: -10, origin_slot: 13, target_id: 24, target_slot: 0, type: "COMBO" },
            { id: 225, origin_id: 14, origin_slot: 0, target_id: 10, target_slot: 0, type: "LATENT" },
            { id: 226, origin_id: 14, origin_slot: 0, target_id: 23, target_slot: 0, type: "LATENT" },
            { id: 227, origin_id: -10, origin_slot: 11, target_id: 13, target_slot: 0, type: "COMBO" },
            { id: 229, origin_id: -10, origin_slot: 8, target_id: 113, target_slot: 0, type: "MODEL" },
            { id: 230, origin_id: -10, origin_slot: 9, target_id: 113, target_slot: 1, type: "COMBO" },
            { id: 231, origin_id: -10, origin_slot: 10, target_id: 113, target_slot: 2, type: "BOOLEAN" },
            { id: 232, origin_id: -10, origin_slot: 6, target_id: 9, target_slot: 2, type: "INT" },
        ],
        extra: { ue_links: [], links_added_by_ue: [] },
    };
}

function createOuterNodeConfig(definition, nodeId) {
    return {
        id: nodeId,
        type: definition.id,
        size: [480, 690],
        flags: {},
        order: 0,
        mode: 0,
        inputs: [
            { name: "first_frame", shape: 7, type: "IMAGE", link: null, tooltip: IMAGE_TOOLTIPS_ZH.first_frame },
            { name: "last_frame", shape: 7, type: "IMAGE", link: null, tooltip: IMAGE_TOOLTIPS_ZH.last_frame },
            { name: "prompt", type: "STRING", widget: { name: "prompt" }, link: null, tooltip: IMAGE_TOOLTIPS_ZH.prompt },
            { name: "width", type: "INT", widget: { name: "width" }, link: null, tooltip: IMAGE_TOOLTIPS_ZH.width },
            { name: "height", type: "INT", widget: { name: "height" }, link: null, tooltip: IMAGE_TOOLTIPS_ZH.height },
            { label: "duration", name: "value_1", type: "FLOAT", widget: { name: "value_1" }, link: null, tooltip: IMAGE_TOOLTIPS_ZH.value_1 },
            { name: "steps", type: "INT", widget: { name: "steps" }, link: null, tooltip: IMAGE_TOOLTIPS_ZH.steps },
            { name: "noise_seed", type: "INT", widget: { name: "noise_seed" }, link: null, tooltip: IMAGE_TOOLTIPS_ZH.noise_seed },
            { name: "unet_name", type: "MODEL", link: null, tooltip: IMAGE_TOOLTIPS_ZH.unet_name },
            { name: "cache_device", type: "COMBO", widget: { name: "cache_device" }, link: null, tooltip: IMAGE_TOOLTIPS_ZH.cache_device },
            { name: "verbose", type: "BOOLEAN", widget: { name: "verbose" }, link: null, tooltip: IMAGE_TOOLTIPS_ZH.verbose },
            { name: "clip_name", type: "COMBO", widget: { name: "clip_name" }, link: null, tooltip: IMAGE_TOOLTIPS_ZH.clip_name },
            { name: "vae_name", type: "COMBO", widget: { name: "vae_name" }, link: null, tooltip: IMAGE_TOOLTIPS_ZH.vae_name },
            { label: "audio_vae", name: "vae_name_1", type: "COMBO", widget: { name: "vae_name_1" }, link: null, tooltip: IMAGE_TOOLTIPS_ZH.vae_name_1 },
        ],
        outputs: [{ localized_name: "VIDEO", name: "VIDEO", type: "VIDEO", links: null }],
        properties: {
            cnr_id: "comfy-core",
            ver: "0.30.0",
            previewExposures: [],
            ue_properties: {
                widget_ue_connectable: {
                    width: true,
                    height: true,
                    value_1: true,
                    steps: true,
                    vae_name_1: true,
                },
                version: "7.8",
                input_ue_unconnectable: {},
            },
        },
        widgets_values: [
            DEFAULT_PROMPT,
            1344,
            768,
            30,
            20,
            DEFAULT_SEED,
            "auto",
            false,
            "minimax-H3\\qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
            "minimax-H3\\minimax_h3_video_vae_fp16.safetensors",
            "minimax-H3\\minimax_h3_audio_vae_fp32.safetensors",
        ],
    };
}

function imageTooltipNodeDef(nodeType) {
    const spec = (type, name, extra = {}) => [type, {
        ...extra,
        tooltip: IMAGE_TOOLTIPS_ZH[name],
    }];
    return {
        name: nodeType,
        display_name: SUBGRAPH_NAME,
        description: "带 MiniMax H3 加速 MODEL 接口、采样步数和缓存运行选项的原生图生视频子图。",
        category: "MiniMaxH3",
        input: {
            required: {
                prompt: spec("STRING", "prompt", { multiline: true }),
                width: spec("INT", "width", { default: 1344, min: 32, max: 16384, step: 32 }),
                height: spec("INT", "height", { default: 768, min: 32, max: 16384, step: 32 }),
                value_1: spec("FLOAT", "value_1", { default: 30, min: 0.1, step: 0.1 }),
                steps: spec("INT", "steps", { default: 20, min: 1, max: 10000, step: 1 }),
                noise_seed: spec("INT", "noise_seed", { default: DEFAULT_SEED, min: 0, step: 1 }),
                unet_name: spec("MODEL", "unet_name"),
                cache_device: [["auto", "gpu", "cpu"], { default: "auto", tooltip: IMAGE_TOOLTIPS_ZH.cache_device }],
                verbose: spec("BOOLEAN", "verbose", { default: false }),
                clip_name: spec("STRING", "clip_name"),
                vae_name: spec("STRING", "vae_name"),
                vae_name_1: spec("STRING", "vae_name_1"),
            },
            optional: {
                first_frame: spec("IMAGE", "first_frame"),
                last_frame: spec("IMAGE", "last_frame"),
            },
        },
        input_order: {
            required: [
                "prompt", "width", "height", "value_1", "steps", "noise_seed",
                "unet_name", "cache_device", "verbose", "clip_name", "vae_name", "vae_name_1",
            ],
            optional: ["first_frame", "last_frame"],
        },
        output: ["VIDEO"],
        output_name: ["VIDEO"],
        output_is_list: [false],
        output_node: false,
        python_module: "custom_nodes.comfyui-speed-minimaxH3",
    };
}

function registerImageTooltipNodeDef(node) {
    const nodeType = node?.type;
    if (!nodeType || node.__linjianTooltipNodeDefRegistered) return;
    node.__linjianTooltipNodeDefRegistered = true;
    imageNodeDefUpdateChain = imageNodeDefUpdateChain
        .then(async () => {
            const nodeDefs = await app.getNodeDefs();
            nodeDefs[nodeType] = imageTooltipNodeDef(nodeType);
            app.updateVueAppNodeDefs(nodeDefs);
        })
        .catch((error) => {
            node.__linjianTooltipNodeDefRegistered = false;
            console.error("Failed to register linjian subgraph tooltip metadata", error);
        });
}

function applyImageTooltips(node) {
    if (!node) return;
    // Current ComfyUI resolves socket hover help from constructor.nodeData,
    // while widget hover help resolves from the live widget. Native subgraph
    // node types have a generated UUID and therefore no backend nodeData of
    // their own, so populate both places explicitly.
    const constructorData = (node.constructor.nodeData ??= {});
    const constructorInputs = (constructorData.inputs ??= {});
    registerImageTooltipNodeDef(node);
    for (const nodeInput of node.inputs ?? []) {
        const tooltip = IMAGE_TOOLTIPS_ZH[nodeInput.name];
        if (!tooltip) continue;
        nodeInput.tooltip = tooltip;
        constructorInputs[nodeInput.name] ??= {};
        constructorInputs[nodeInput.name].tooltip = tooltip;
    }
    for (const widget of node.widgets ?? []) {
        const tooltip = IMAGE_TOOLTIPS_ZH[widget.name];
        if (!tooltip) continue;
        widget.tooltip = tooltip;
        widget.options ??= {};
        widget.options.tooltip = tooltip;
    }
}

function isLinjianImageSubgraphNode(node) {
    return (
        node?.title === SUBGRAPH_NAME
        || node?.subgraph?.name === SUBGRAPH_NAME
        || node?.constructor?.title === SUBGRAPH_NAME
    );
}

function graphLink(graph, linkId) {
    if (linkId == null) return null;
    return (
        graph?.getLink?.(linkId)
        ?? graph?.links?.get?.(linkId)
        ?? graph?._links?.get?.(linkId)
        ?? graph?.links?.[linkId]
        ?? graph?._links?.[linkId]
        ?? null
    );
}

function instantiateImageSubgraph(graph, position) {
    const definition = createSubgraphDefinition();
    const subgraph = app.rootGraph.createSubgraph(definition);
    subgraph.configure(definition);
    for (const innerNode of subgraph.nodes) {
        innerNode.onGraphConfigured?.();
    }
    for (const innerNode of subgraph.nodes) {
        innerNode.onAfterGraphConfigured?.();
    }
    subgraph.inputNode.arrange();
    subgraph.outputNode.arrange();

    const replacement = globalThis.LiteGraph.createNode(definition.id);
    if (!replacement) throw new Error("ComfyUI did not register the bundled subgraph node type");

    replacement.pos = [...position];
    graph.add(replacement);
    const outerConfig = createOuterNodeConfig(definition, replacement.id);
    replacement.configure(outerConfig);
    globalThis.LiteGraph.LGraph.proxyWidgetMigrationFlush?.(replacement, outerConfig);
    globalThis.LiteGraph.LGraph.autoExposePreviewNodes?.(replacement);
    applyImageTooltips(replacement);
    replacement.setSize?.([480, 690]);
    return replacement;
}

function migrateLegacyImageNode(legacyNode) {
    const graph = legacyNode?.graph;
    if (
        !graph
        || legacyNode.__linjianMigrating
        || !isLinjianImageSubgraphNode(legacyNode)
        || legacyNode.inputs?.some((item) => item.name === "steps")
    ) return;

    legacyNode.__linjianMigrating = true;
    try {
        const widgetValues = new Map(
            (legacyNode.widgets ?? []).map((widget) => [widget.name, widget.value]),
        );
        const incoming = [];
        for (const oldInput of legacyNode.inputs ?? []) {
            const link = graphLink(graph, oldInput.link);
            const originNode = link ? graph.getNodeById?.(link.origin_id) : null;
            if (originNode) {
                incoming.push({
                    originNode,
                    originSlot: link.origin_slot,
                    inputName: oldInput.name,
                });
            }
        }

        const outgoing = [];
        for (const oldOutput of legacyNode.outputs ?? []) {
            for (const linkId of oldOutput.links ?? []) {
                const link = graphLink(graph, linkId);
                const targetNode = link ? graph.getNodeById?.(link.target_id) : null;
                if (targetNode) {
                    outgoing.push({
                        outputName: oldOutput.name,
                        targetNode,
                        targetSlot: link.target_slot,
                    });
                }
            }
        }

        const replacement = instantiateImageSubgraph(graph, legacyNode.pos ?? [0, 0]);
        for (const widget of replacement.widgets ?? []) {
            if (widgetValues.has(widget.name)) widget.value = widgetValues.get(widget.name);
        }

        graph.remove(legacyNode);
        for (const saved of incoming) {
            const targetSlot = replacement.inputs?.findIndex(
                (item) => item.name === saved.inputName,
            );
            if (targetSlot >= 0) saved.originNode.connect(saved.originSlot, replacement, targetSlot);
        }
        for (const saved of outgoing) {
            const originSlot = replacement.outputs?.findIndex(
                (item) => item.name === saved.outputName,
            );
            if (originSlot >= 0) replacement.connect(originSlot, saved.targetNode, saved.targetSlot);
        }

        app.canvas?.setDirty?.(true, true);
        console.info(
            "Updated legacy linjian Image to Video node with steps, cache_device, verbose and Chinese tooltips.",
        );
    } catch (error) {
        legacyNode.__linjianMigrating = false;
        console.error("Failed to update legacy linjian MiniMax H3 subgraph", error);
    }
}

function replaceBuilderNode(builder) {
    const graph = builder.graph;
    if (!graph || builder.__linjianReplacing) return;
    builder.__linjianReplacing = true;

    try {
        const replacement = instantiateImageSubgraph(graph, builder.pos);
        graph.remove(builder);
        app.canvas?.select?.(replacement);
        app.canvas?.setDirty?.(true, true);
    } catch (error) {
        builder.__linjianReplacing = false;
        console.error("Failed to create linjian MiniMax H3 subgraph", error);
    }
}

app.registerExtension({
    name: "comfyui-speed-minimaxH3.linjian-subgraph",
    nodeCreated(node) {
        if (node.comfyClass === BUILDER_NODE) {
            queueMicrotask(() => replaceBuilderNode(node));
        } else if (isLinjianImageSubgraphNode(node)) {
            queueMicrotask(() => applyImageTooltips(node));
        }
    },
    loadedGraphNode(node) {
        if (!isLinjianImageSubgraphNode(node)) return;
        queueMicrotask(() => {
            applyImageTooltips(node);
            migrateLegacyImageNode(node);
        });
    },
});

export {
    IMAGE_TOOLTIPS_ZH,
    createOuterNodeConfig,
    createSubgraphDefinition,
    migrateLegacyImageNode,
};
