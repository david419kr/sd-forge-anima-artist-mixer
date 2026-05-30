from __future__ import annotations

import logging
import math
import re
from contextlib import nullcontext
from typing import Any

import gradio as gr
import torch
import torch.nn as nn

import modules.scripts as scripts
from modules import script_callbacks, shared

logger = logging.getLogger(__name__)

TITLE = "AnimaArtistCrossAttn"
SECTION = ("anima_artist_mixer", "Anima Artist Mixer")

OPT_POSITION = "aam_artist_chain_position"
OPT_STRENGTH_IN_CFG_ROW = "aam_artist_strength_in_cfg_row"
POSITION_ABOVE = "above"
POSITION_BETWEEN = "between"
POSITION_BELOW = "below"
POSITION_CHOICES = [POSITION_ABOVE, POSITION_BETWEEN, POSITION_BELOW]

FUSION_INTERPOLATE = "interpolate"
FUSION_CONCAT_WITH_BASE = "concat_with_base"
FUSION_BASE_PRESERVE = "base_preserve"
FUSION_CHOICES = [FUSION_INTERPOLATE, FUSION_CONCAT_WITH_BASE, FUSION_BASE_PRESERVE]

COMBINE_CONCAT = "concat"
COMBINE_OUTPUT_AVG = "output_avg"
COMBINE_LOWRANK_AVG = "lowrank_avg"
COMBINE_CHOICES = [COMBINE_OUTPUT_AVG, COMBINE_CONCAT, COMBINE_LOWRANK_AVG]

MAX_ARTISTS = 32
STATIC_CAPTURE_K_DEFAULT = 6
STATIC_CAPTURE_K_MAX = 12
AAM_TOPROW_PATCH_VERSION = 2
_EXTRA_NETWORK_RE = re.compile(r"<[^:>]+:[^>]+>")

_ARTIST_COMPONENTS: dict[str, gr.components.Component] = {}


def _register_settings() -> None:
    shared.opts.add_option(
        OPT_POSITION,
        shared.OptionInfo(
            POSITION_BETWEEN,
            "Artist Chain textbox position",
            gr.Radio,
            {"choices": POSITION_CHOICES},
            section=SECTION,
        )
        .info("Requires Reload UI.")
        .needs_reload_ui(),
    )
    shared.opts.add_option(
        OPT_STRENGTH_IN_CFG_ROW,
        shared.OptionInfo(
            False,
            "Show Artist Strength in the CFG Scale row",
            gr.Checkbox,
            section=SECTION,
        )
        .info("Moves the Strength slider out of the AnimaArtistCrossAttn accordion. Requires Reload UI.")
        .needs_reload_ui(),
    )


script_callbacks.on_ui_settings(_register_settings)


def _artist_position() -> str:
    value = getattr(shared.opts, OPT_POSITION, POSITION_BETWEEN)
    return value if value in POSITION_CHOICES else POSITION_BETWEEN


def _artist_strength_in_cfg_row() -> bool:
    return bool(getattr(shared.opts, OPT_STRENGTH_IN_CFG_ROW, False))


def _create_strength_slider(
    id_part: str,
    *,
    label: str,
    visible: bool = True,
    elem_id: str | None = None,
    scale: int | None = None,
) -> gr.Slider:
    kwargs: dict[str, Any] = {
        "minimum": 0.0,
        "maximum": 4.0,
        "value": 1.0,
        "step": 0.05,
        "label": label,
        "visible": visible,
    }
    if elem_id is not None:
        kwargs["elem_id"] = elem_id
    if scale is not None:
        kwargs["scale"] = scale
    return gr.Slider(**kwargs)


def _toprow_artist_components() -> dict[str, gr.components.Component]:
    from modules.ui_toprow import Toprow

    registry = getattr(Toprow, "_aam_artist_components", None)
    if not isinstance(registry, dict):
        registry = {}
        Toprow._aam_artist_components = registry
    return registry


def _remember_artist_chain(id_part: str, artist_chain: gr.Textbox) -> None:
    _ARTIST_COMPONENTS[id_part] = artist_chain
    _toprow_artist_components()[id_part] = artist_chain


def _artist_chain_component(id_part: str) -> gr.components.Component | None:
    component = _ARTIST_COMPONENTS.get(id_part)
    if component is not None:
        return component
    return _toprow_artist_components().get(id_part)


def _create_artist_chain_row(toprow: Any) -> gr.Textbox:
    id_part = toprow.id_part
    with gr.Row(
        elem_id=f"{id_part}_anima_artist_chain_row",
        elem_classes=["prompt-row", "anima-artist-chain-row"],
    ):
        artist_chain = gr.Textbox(
            label="Artist Chain",
            elem_id=f"{id_part}_anima_artist_chain",
            show_label=False,
            lines=1,
            max_lines=1,
            placeholder="Artist Chain",
            elem_classes=["prompt", "anima-artist-chain"],
        )

    toprow.anima_artist_chain = artist_chain
    _remember_artist_chain(id_part, artist_chain)
    return artist_chain


def _patch_toprow() -> None:
    from modules.ui_toprow import Toprow

    current_create_prompts = Toprow.create_prompts
    if (
        getattr(current_create_prompts, "_aam_patch_version", 0) >= AAM_TOPROW_PATCH_VERSION
        and getattr(current_create_prompts, "_aam_uses_shared_registry", False)
    ):
        return
    original_create_prompts = getattr(current_create_prompts, "_aam_original", current_create_prompts)

    def create_prompts(self):
        with gr.Column(
            elem_id=f"{self.id_part}_prompt_container",
            elem_classes=self._container_class(),
            scale=6,
        ):
            container = (
                gr.Accordion(label="Prompts", open=False)
                if shared.opts.prompt_box_style == "Accordion"
                else nullcontext()
            )
            container.__enter__()

            position = _artist_position()
            if position == POSITION_ABOVE:
                _create_artist_chain_row(self)

            with gr.Row(elem_id=f"{self.id_part}_prompt_row", elem_classes=["prompt-row"]):
                self.prompt = gr.Textbox(
                    label="Prompt",
                    elem_id=f"{self.id_part}_prompt",
                    show_label=False,
                    lines=3,
                    placeholder="Prompt\n(Ctrl+Enter to Generate ; Alt+Enter to Skip ; Esc to Interrupt)",
                    elem_classes=["prompt"],
                )
                self.prompt_img = gr.File(
                    elem_id=f"{self.id_part}_prompt_image",
                    file_count="single",
                    type="binary",
                    visible=False,
                )

            if position == POSITION_BETWEEN:
                _create_artist_chain_row(self)

            with gr.Row(elem_id=f"{self.id_part}_neg_prompt_row", elem_classes=["prompt-row"]):
                self.negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    elem_id=f"{self.id_part}_neg_prompt",
                    show_label=False,
                    lines=3,
                    placeholder="Negative Prompt\n(Ctrl+Enter to Generate ; Alt+Enter to Skip ; Esc to Interrupt)",
                    elem_classes=["prompt"],
                )

            if position == POSITION_BELOW:
                _create_artist_chain_row(self)

            container.__exit__(None, None, None)

        from modules import images as images_module

        self.prompt_img.change(
            fn=images_module.image_data,
            inputs=[self.prompt_img],
            outputs=[self.prompt, self.prompt_img],
            show_progress=False,
        )

    create_prompts._aam_patched = True
    create_prompts._aam_original = original_create_prompts
    create_prompts._aam_patch_version = AAM_TOPROW_PATCH_VERSION
    create_prompts._aam_uses_shared_registry = True
    Toprow.create_prompts = create_prompts


script_callbacks.on_before_ui(_patch_toprow)


def _split_artist_chain(chain: str | None) -> list[str]:
    if not chain:
        return []

    normalized = str(chain).replace("，", ",").replace("\n", ",").replace("\r", ",")
    parts = [_EXTRA_NETWORK_RE.sub("", part).strip() for part in normalized.split(",")]
    return [part for part in parts if part]


def _parse_artist_weights(parts: list[str]) -> tuple[list[str], list[float], bool]:
    names: list[str] = []
    weights: list[float] = []
    has_explicit = False

    for raw in parts:
        text = str(raw or "").strip()
        if not text:
            continue

        weight = 1.0
        explicit = False
        if "::" in text:
            head = text[2:] if text.startswith("::") else text
            if "::" in head:
                name_part, _, weight_part = head.rpartition("::")
                try:
                    weight = max(0.0, min(4.0, float(weight_part.strip())))
                    text = name_part.strip()
                    explicit = True
                except ValueError:
                    pass

        if not text:
            continue
        names.append(text)
        weights.append(weight)
        has_explicit = has_explicit or explicit

    return names, weights, has_explicit


def _parse_layer_filter(text: str | None, num_blocks: int) -> list[int] | None:
    if not text:
        return None

    value = str(text).replace("，", ",").replace(" ", "")
    if not value:
        return None

    result: set[int] = set()
    for part in value.split(","):
        if not part:
            continue

        if "-" in part[1:]:
            dash_idx = part.index("-", 1)
            try:
                lo = int(part[:dash_idx])
                hi = int(part[dash_idx + 1 :])
            except ValueError:
                continue

            if lo < 0:
                lo += num_blocks
            if hi < 0:
                hi += num_blocks
            if lo > hi:
                lo, hi = hi, lo
            lo = max(0, lo)
            hi = min(num_blocks - 1, hi)
            if lo <= hi:
                result.update(range(lo, hi + 1))
        else:
            try:
                index = int(part)
            except ValueError:
                continue

            if index < 0:
                index += num_blocks
            if 0 <= index < num_blocks:
                result.add(index)

    return sorted(result) if result else None


def _normalize_weights(weights: list[float]) -> list[float]:
    total = sum(abs(weight) for weight in weights)
    if total <= 1e-8:
        return [1.0 / len(weights)] * len(weights)
    return [weight / total for weight in weights]


def _clamp_strength(value: float) -> float:
    return max(0.0, min(4.0, float(value)))


def _project_perpendicular(delta: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    base_f = base.float()
    delta_f = delta.float()
    denom = (base_f * base_f).sum(dim=-1, keepdim=True).clamp_min(1e-8)
    parallel = ((delta_f * base_f).sum(dim=-1, keepdim=True) / denom) * base_f
    return (delta_f - parallel).to(dtype=delta.dtype, device=delta.device)


def _unwrap_cross_attn(cross_attn: nn.Module) -> nn.Module:
    while isinstance(cross_attn, _CrossAttnWrapper):
        cross_attn = cross_attn.original
    return cross_attn


def _validate_diffusion_model(diffusion_model: nn.Module) -> tuple[bool, int, str]:
    blocks = getattr(diffusion_model, "blocks", None)
    if blocks is None:
        return False, 0, f"{type(diffusion_model).__name__} has no .blocks"
    if len(blocks) == 0:
        return False, 0, ".blocks is empty"
    if not hasattr(blocks[0], "cross_attn"):
        return False, 0, "blocks[0] has no cross_attn"

    cross_attn = _unwrap_cross_attn(blocks[0].cross_attn)
    if not hasattr(cross_attn, "context_dim"):
        return False, 0, "cross_attn has no context_dim"

    return True, len(blocks), "ok"


def _as_tensor(
    conditioning: Any,
    keys: tuple[str, ...] = ("crossattn", "cross_attn"),
) -> torch.Tensor | None:
    if torch.is_tensor(conditioning):
        return conditioning
    if isinstance(conditioning, dict):
        value = None
        for key in keys:
            value = conditioning.get(key)
            if value is not None:
                break
        return value if torch.is_tensor(value) else None
    return None


def _squeeze_single_conditioning(tensor: torch.Tensor) -> torch.Tensor:
    result = tensor
    while result.dim() > 2 and result.shape[0] == 1:
        result = result.squeeze(0)
    return result


def _pad_sequence_tensor(tensor: torch.Tensor, target_tokens: int) -> torch.Tensor:
    if tensor.shape[-2] >= target_tokens:
        return tensor

    pad_count = target_tokens - tensor.shape[-2]
    tail = tensor[..., -1:, :].repeat(*([1] * (tensor.dim() - 2)), pad_count, 1)
    return torch.cat([tensor, tail], dim=-2)


def _conditioning_to_batch_tensor(
    conditioning: Any,
    expected_batch: int,
    keys: tuple[str, ...] = ("crossattn", "cross_attn"),
) -> torch.Tensor | None:
    if isinstance(conditioning, (list, tuple)):
        items = [_as_tensor(item, keys) for item in conditioning]
        items = [item for item in items if item is not None]
        if not items:
            return None

        squeezed = [_squeeze_single_conditioning(item) for item in items]
        max_tokens = max(item.shape[-2] for item in squeezed)
        padded = [_pad_sequence_tensor(item, max_tokens) for item in squeezed]
        stacked = torch.stack(padded, dim=0)
        if stacked.dim() == 4 and stacked.shape[1] == 1:
            stacked = stacked.squeeze(1)
        return stacked

    tensor = _as_tensor(conditioning, keys)
    if tensor is None:
        return None

    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 4 and tensor.shape[1] == 1:
        tensor = tensor.squeeze(1)

    if expected_batch > 0 and tensor.shape[0] != expected_batch:
        tensor = _broadcast_batch(tensor, expected_batch)

    return tensor


def _match_context_rank(artist: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    result = artist
    if result.dim() == 2:
        result = result.unsqueeze(0)

    if context.dim() == 4 and result.dim() == 3:
        result = result.unsqueeze(1)
    elif context.dim() == 3 and result.dim() == 4 and result.shape[1] == 1:
        result = result.squeeze(1)

    return result


def _fit_token_length(
    tensor: torch.Tensor,
    token_length: int,
    pad_value: float = 1.0,
) -> torch.Tensor:
    if tensor.shape[-2] == token_length:
        return tensor
    if tensor.shape[-2] > token_length:
        return tensor[..., :token_length, :]

    pad_shape = (*tensor.shape[:-2], token_length - tensor.shape[-2], tensor.shape[-1])
    padding = torch.full(pad_shape, pad_value, device=tensor.device, dtype=tensor.dtype)
    return torch.cat([tensor, padding], dim=-2)


def _fit_mask_to_context(mask: torch.Tensor | None, context: torch.Tensor) -> torch.Tensor | None:
    if mask is None:
        return None

    result = mask
    if result.dim() == 1:
        result = result.view(1, -1, 1)
    elif result.dim() == 2:
        result = result.unsqueeze(0)

    result = _match_context_rank(result, context)
    result = _fit_token_length(result, context.shape[-2], pad_value=1.0)
    result = _broadcast_batch(result, context.shape[0])
    return result.to(device=context.device, dtype=context.dtype)


def _ones_mask_like(context: torch.Tensor) -> torch.Tensor:
    return torch.ones(
        (*context.shape[:-1], 1),
        device=context.device,
        dtype=context.dtype,
    )


def _mask_or_ones(mask: torch.Tensor | None, context: torch.Tensor) -> torch.Tensor:
    return mask if mask is not None else _ones_mask_like(context)


def _mask_from_options(transformer_options: dict[str, Any], context: torch.Tensor) -> torch.Tensor | None:
    mask = transformer_options.get("negpip_mask") if isinstance(transformer_options, dict) else None
    if not torch.is_tensor(mask):
        return None
    return _fit_mask_to_context(mask, context)


def _with_negpip_mask(
    transformer_options: dict[str, Any],
    mask: torch.Tensor | None,
) -> dict[str, Any]:
    next_options = dict(transformer_options)
    if mask is None:
        next_options.pop("negpip_mask", None)
    else:
        next_options["negpip_mask"] = mask
    return next_options


def _callable_func(callable_obj: Any) -> Any:
    return getattr(callable_obj, "__func__", callable_obj)


def _closure_map(func: Any) -> dict[str, Any]:
    code = getattr(func, "__code__", None)
    closure = getattr(func, "__closure__", None)
    if code is None or closure is None:
        return {}
    return {
        name: cell.cell_contents
        for name, cell in zip(code.co_freevars, closure)
    }


def _extract_forge_couple_state(attn_module: nn.Module) -> dict[str, Any] | None:
    forward = _callable_func(getattr(attn_module, "forward", None))
    if forward is None or not getattr(forward, "_couple", False):
        return None

    state = _closure_map(forward)
    inner = state.get("func")
    if callable(inner):
        inner_state = _closure_map(inner)
        if "mask" in inner_state and "num_conds" in inner_state:
            state = inner_state

    required = {"mask", "num_conds", "width", "height", "dit"}
    if not required.issubset(state):
        return None

    original_forward = getattr(attn_module, "couple_orig_forward", None)
    if original_forward is None:
        return None

    state["original_forward"] = original_forward
    return state


def _zero_pad_token_length(tensor: torch.Tensor, target_tokens: int) -> torch.Tensor:
    if tensor.shape[-2] >= target_tokens:
        return tensor[..., :target_tokens, :]

    pad_shape = (*tensor.shape[:-2], target_tokens - tensor.shape[-2], tensor.shape[-1])
    padding = torch.zeros(pad_shape, device=tensor.device, dtype=tensor.dtype)
    return torch.cat([tensor, padding], dim=-2)


def _pad_to_token_multiple(tensor: torch.Tensor, multiple: int = 512) -> torch.Tensor:
    target_tokens = max(multiple, math.ceil(tensor.shape[-2] / multiple) * multiple)
    return _zero_pad_token_length(tensor, target_tokens)


def _fit_tokens_for_couple(tensor: torch.Tensor, target_tokens: int, pad_value: float = 0.0) -> torch.Tensor:
    token_count = tensor.shape[-2]
    if token_count == target_tokens:
        return tensor
    if token_count > 0 and target_tokens % token_count == 0:
        return tensor.repeat(*([1] * (tensor.dim() - 2)), target_tokens // token_count, 1)
    if token_count < target_tokens:
        pad_shape = (*tensor.shape[:-2], target_tokens - token_count, tensor.shape[-1])
        padding = torch.full(pad_shape, pad_value, device=tensor.device, dtype=tensor.dtype)
        return torch.cat([tensor, padding], dim=-2)
    return tensor[..., :target_tokens, :]


def _lcm_for_values(values: list[int]) -> int:
    result = 1
    for value in values:
        if value <= 0:
            continue
        result = result * value // math.gcd(result, value)
    return result


def _forge_couple_region_conds(
    state: dict[str, Any],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor] | None:
    conds = state.get("conds")
    num_conds = int(state.get("num_conds", 0))
    region_count = num_conds - 1
    if region_count <= 0 or not isinstance(conds, list) or len(conds) < region_count:
        return None

    result: list[torch.Tensor] = []
    for cond in conds[:region_count]:
        if not torch.is_tensor(cond):
            return None
        region = cond.to(device=device, dtype=dtype)
        if region.dim() == 2:
            region = region.unsqueeze(0)
        if region.dim() == 4 and region.shape[1] == 1:
            region = region.squeeze(1)
        if region.dim() != 3:
            return None
        result.append(_pad_to_token_multiple(_broadcast_batch(region, batch_size)))

    return result


def _forge_couple_original_region_masks(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor | None]:
    try:
        import scripts.negpip as negpip_module
    except Exception:
        return []

    hook_state = getattr(negpip_module, "ANIMA_HOOK_STATE", {}) or {}
    masks = hook_state.get("forge_couple_negpip_masks") or []
    result: list[torch.Tensor | None] = []
    for mask in masks:
        if not torch.is_tensor(mask):
            result.append(None)
            continue

        fitted = mask.to(device=device, dtype=dtype)
        if fitted.dim() == 1:
            fitted = fitted.view(1, -1, 1)
        elif fitted.dim() == 2:
            fitted = fitted.unsqueeze(0)
        if fitted.dim() == 4 and fitted.shape[1] == 1:
            fitted = fitted.squeeze(1)
        if fitted.dim() != 3:
            result.append(None)
            continue
        result.append(_broadcast_batch(fitted, batch_size))

    return result


def _merge_forge_couple_region_context(
    region: torch.Tensor,
    artist: torch.Tensor,
    region_mask: torch.Tensor | None,
    artist_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    region = _pad_to_token_multiple(region)
    artist = _pad_to_token_multiple(artist)
    region_mask = _fit_mask_to_context(region_mask, region) if region_mask is not None else None
    artist_mask = _fit_mask_to_context(artist_mask, artist) if artist_mask is not None else None

    if region.shape[-2] + artist.shape[-2] <= 1024:
        merged = _cat_context(artist, region)
        if region_mask is None and artist_mask is None:
            return merged, None
        return merged, _cat_context(
            _mask_or_ones(artist_mask, artist),
            _mask_or_ones(region_mask, region),
        )

    artist_fit = _fit_tokens_for_couple(artist, 512)
    region_fit = _fit_tokens_for_couple(region, 512)
    merged = _cat_context(artist_fit, region_fit)

    if region_mask is None and artist_mask is None:
        return merged, None

    artist_mask_fit = _fit_tokens_for_couple(_mask_or_ones(artist_mask, artist), 512, pad_value=1.0)
    region_mask_fit = _fit_tokens_for_couple(_mask_or_ones(region_mask, region), 512, pad_value=1.0)
    return merged, _cat_context(artist_mask_fit, region_mask_fit)


def _forge_couple_artist_forward(
    attn_module: nn.Module,
    x: torch.Tensor,
    context: torch.Tensor,
    rope_emb: torch.Tensor | None,
    transformer_options: dict[str, Any],
) -> torch.Tensor | None:
    state = _extract_forge_couple_state(attn_module)
    cond_or_unconds = transformer_options.get("cond_or_uncond") if transformer_options else None
    if state is None or context is None or not cond_or_unconds:
        return None

    num_chunks = len(cond_or_unconds)
    if num_chunks <= 0 or x.shape[0] % num_chunks != 0:
        return None

    batch_size = x.shape[0] // num_chunks
    context_3d = context.squeeze(1) if context.dim() == 4 and context.shape[1] == 1 else context
    if context_3d.dim() != 3 or context_3d.shape[0] != x.shape[0]:
        return None

    try:
        from lib_couple.attention_masks import get_dit_mask
    except Exception:
        return None

    original_forward = state["original_forward"]
    num_conds = int(state["num_conds"])
    mask = state["mask"].to(device=x.device, dtype=x.dtype)
    dit = state["dit"]
    width = int(state["width"])
    height = int(state["height"])

    x_chunks = x.chunk(num_chunks, dim=0)
    context_3d = _pad_to_token_multiple(context_3d)
    context_chunks = context_3d.chunk(num_chunks, dim=0)

    negpip_mask = _mask_from_options(transformer_options, context_3d)
    negpip_chunks = negpip_mask.chunk(num_chunks, dim=0) if negpip_mask is not None else None
    region_conds = _forge_couple_region_conds(state, batch_size, context_3d.device, context_3d.dtype)
    original_region_masks = _forge_couple_original_region_masks(
        batch_size,
        context_3d.device,
        context_3d.dtype,
    )
    couple_lcm_tokens = None
    if region_conds is not None:
        artist_tokens = context_3d.shape[-2]
        region_slot_lengths = []
        for region_cond in region_conds:
            region_tokens = region_cond.shape[-2]
            if region_tokens + artist_tokens <= 1024:
                region_slot_lengths.append(region_tokens + artist_tokens)
            else:
                region_slot_lengths.append(1024)

        couple_lcm_tokens = _lcm_for_values([artist_tokens] + region_slot_lengths)
        if couple_lcm_tokens not in (512, 1024):
            region_conds = None
            couple_lcm_tokens = None

    new_x = []
    new_context = []
    new_negpip_masks = []

    for idx, cond_or_uncond in enumerate(cond_or_unconds):
        c_target = context_chunks[idx]
        if cond_or_uncond == 1:
            new_x.append(x_chunks[idx])
            if couple_lcm_tokens is None:
                new_context.append(c_target)
                if negpip_chunks is not None:
                    new_negpip_masks.append(negpip_chunks[idx])
            else:
                c_target_fit = _fit_tokens_for_couple(c_target, couple_lcm_tokens)
                new_context.append(c_target_fit)
                if negpip_chunks is not None:
                    new_negpip_masks.append(_fit_tokens_for_couple(negpip_chunks[idx], couple_lcm_tokens, pad_value=1.0))
        else:
            new_x.append(x_chunks[idx].repeat(num_conds, *([1] * (x.dim() - 1))))

            if region_conds is None or couple_lcm_tokens is None:
                new_context.append(c_target.repeat(num_conds, *([1] * (c_target.dim() - 1))))
                if negpip_chunks is not None:
                    new_negpip_masks.append(negpip_chunks[idx].repeat(num_conds, *([1] * (negpip_chunks[idx].dim() - 1))))
                continue

            artist_mask = negpip_chunks[idx] if negpip_chunks is not None else None
            region_slots = []
            region_slot_masks = []
            for region_index, region_cond in enumerate(region_conds):
                region_mask = original_region_masks[region_index] if region_index < len(original_region_masks) else None
                slot, slot_mask = _merge_forge_couple_region_context(
                    region_cond,
                    c_target,
                    region_mask,
                    artist_mask,
                )
                region_slots.append(slot)
                region_slot_masks.append(slot_mask)

            c_target_fit = _fit_tokens_for_couple(c_target, couple_lcm_tokens)
            fitted_slots = [_fit_tokens_for_couple(slot, couple_lcm_tokens) for slot in region_slots]
            new_context.append(torch.cat([c_target_fit] + fitted_slots, dim=0))

            if negpip_chunks is not None or any(slot_mask is not None for slot_mask in region_slot_masks):
                base_mask = (
                    _fit_tokens_for_couple(artist_mask, couple_lcm_tokens, pad_value=1.0)
                    if artist_mask is not None
                    else _ones_mask_like(c_target_fit)
                )
                fitted_slot_masks = [
                    _fit_tokens_for_couple(slot_mask, couple_lcm_tokens, pad_value=1.0)
                    if slot_mask is not None
                    else _ones_mask_like(slot)
                    for slot, slot_mask in zip(fitted_slots, region_slot_masks)
                ]
                new_negpip_masks.append(torch.cat([base_mask] + fitted_slot_masks, dim=0))

    x_in = torch.cat(new_x, dim=0)
    ctx_in = torch.cat(new_context, dim=0).to(dtype=x_in.dtype)

    next_options = dict(transformer_options)
    if new_negpip_masks:
        next_options["negpip_mask"] = torch.cat(new_negpip_masks, dim=0).to(device=ctx_in.device, dtype=ctx_in.dtype)

    out = original_forward(
        x_in,
        context=ctx_in,
        rope_emb=rope_emb,
        transformer_options=next_options,
    )

    seq_len = int(out.shape[1])
    patch_size = int(getattr(dit, "patch_spatial", 2))
    mask_downsample = get_dit_mask(mask, seq_len, width, height, patch_size=patch_size).to(out)

    outputs = []
    pos = 0
    for cond_or_uncond in cond_or_unconds:
        if cond_or_uncond == 1:
            outputs.append(out[pos : pos + batch_size])
            pos += batch_size
        else:
            chunk = out[pos : pos + num_conds * batch_size]
            chunk = chunk.view(num_conds, batch_size, seq_len, -1)
            outputs.append((chunk * mask_downsample).sum(dim=0))
            pos += num_conds * batch_size

    return torch.cat(outputs, dim=0)


def _broadcast_batch(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.expand(batch_size, *tensor.shape[1:])
    if batch_size % tensor.shape[0] == 0:
        return tensor.repeat(batch_size // tensor.shape[0], *([1] * (tensor.dim() - 1)))
    return tensor[:1].expand(batch_size, *tensor.shape[1:])


def _cat_context(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.cat([left, right], dim=-2)


def _combine_concat(individuals: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    parts = [artist * float(weight) for artist, weight in zip(individuals, weights)]
    return torch.cat(parts, dim=-2)


def _combine_concat_masks(
    individuals: list[torch.Tensor],
    masks: list[torch.Tensor | None],
) -> torch.Tensor | None:
    if not any(mask is not None for mask in masks):
        return None

    parts = [
        mask if mask is not None else _ones_mask_like(artist)
        for artist, mask in zip(individuals, masks)
    ]
    return torch.cat(parts, dim=-2)


def _resolve_mask(
    cond_or_uncond: list[int] | tuple[int, ...] | None,
    row_count: int,
    apply_to_uncond: bool,
    state: dict[str, Any],
) -> list[bool]:
    markers = list(cond_or_uncond or [])
    if markers and len(markers) != row_count and row_count % len(markers) == 0:
        rows_per_chunk = row_count // len(markers)
        markers = [
            marker
            for marker in markers
            for _ in range(rows_per_chunk)
        ]

    if not markers or len(markers) != row_count:
        if not state.get("_warned_mask", False):
            logger.warning(
                "[AnimaArtistMixer] cond_or_uncond is unavailable (got=%s, batch=%d); applying to all rows.",
                cond_or_uncond,
                row_count,
            )
            state["_warned_mask"] = True
        return [True] * row_count

    if apply_to_uncond:
        return [True] * row_count

    return [marker == 0 for marker in markers]


def _in_sigma_range(state: dict[str, Any]) -> bool:
    sigma_range = state.get("sigma_range")
    if sigma_range is None:
        return True

    current_sigma = state.get("current_sigma")
    if current_sigma is None:
        return True

    lo, hi = sigma_range
    return lo <= current_sigma <= hi


def _shape_tuple(value: Any) -> tuple[int, ...] | None:
    return tuple(value.shape) if torch.is_tensor(value) else None


class _CrossAttnWrapper(nn.Module):
    def __init__(self, original: nn.Module, shared_state: dict[str, Any], layer_idx: int):
        super().__init__()
        self.original = original
        self._state = shared_state
        self._layer_idx = layer_idx
        self._disabled = False
        self._deferred_signature = None
        self._deferred_delta_sum: torch.Tensor | None = None
        self._deferred_count = 0
        self._static_signature = None
        self._static_seen_sigmas: set[float] = set()
        self._static_accumulator: list[torch.Tensor] | None = None
        self._static_count = 0
        self._static_frozen_outputs: list[torch.Tensor] | None = None
        self._static_max_sigma: float | None = None

    def _call_original(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        rope_emb: torch.Tensor | None,
        transformer_options: dict[str, Any],
        *,
        forge_couple_artist: bool = False,
    ) -> torch.Tensor:
        if forge_couple_artist:
            couple_out = _forge_couple_artist_forward(
                self.original,
                x,
                context,
                rope_emb,
                transformer_options,
            )
            if couple_out is not None:
                return couple_out

        return self.original(
            x,
            context,
            rope_emb=rope_emb,
            transformer_options=transformer_options,
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        rope_emb: torch.Tensor | None = None,
        transformer_options: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        state = self._state
        if context is None:
            return self.original(
                x,
                context,
                rope_emb=rope_emb,
                transformer_options=transformer_options,
            )

        if self._disabled or not _in_sigma_range(state):
            return self.original(
                x,
                context,
                rope_emb=rope_emb,
                transformer_options=transformer_options,
            )

        try:
            return self._dispatch(x, context, rope_emb, transformer_options or {})
        except Exception as exc:
            logger.exception(
                "[AnimaArtistMixer] Layer %d artist cross-attn failed; falling back for this layer: %s",
                self._layer_idx,
                exc,
            )
            self._disabled = True
            return self.original(
                x,
                context,
                rope_emb=rope_emb,
                transformer_options=transformer_options,
            )

    def _dispatch(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        rope_emb: torch.Tensor | None,
        transformer_options: dict[str, Any],
    ) -> torch.Tensor:
        state = self._state
        individuals = [
            _match_context_rank(artist, context).to(device=context.device, dtype=context.dtype)
            for artist in state["individuals"]
        ]
        stored_masks = list(state.get("individual_masks") or [])
        if len(stored_masks) < len(individuals):
            stored_masks.extend([None] * (len(individuals) - len(stored_masks)))
        individual_masks = [
            _fit_mask_to_context(mask, artist) if mask is not None else None
            for artist, mask in zip(individuals, stored_masks)
        ]
        weights = state["user_weights"]
        batch_size = context.shape[0]
        cond_or_uncond = transformer_options.get("cond_or_uncond")
        mask = _resolve_mask(cond_or_uncond, batch_size, state["apply_to_uncond"], state)

        if not any(mask):
            return self.original(
                x,
                context,
                rope_emb=rope_emb,
                transformer_options=transformer_options,
            )

        combine_mode = state["combine_mode"]
        if combine_mode == COMBINE_LOWRANK_AVG and len(individuals) >= 2:
            return self._fwd_lowrank_avg(
                x,
                context,
                rope_emb,
                transformer_options,
                individuals,
                individual_masks,
                weights,
                mask,
                state["fusion_mode"],
                float(state["strength"]),
            )

        if combine_mode in (COMBINE_OUTPUT_AVG, COMBINE_LOWRANK_AVG):
            return self._fwd_output_avg(
                x,
                context,
                rope_emb,
                transformer_options,
                individuals,
                individual_masks,
                weights,
                mask,
                state["fusion_mode"],
                float(state["strength"]),
            )

        combined = _combine_concat(individuals, weights)
        combined_mask = _combine_concat_masks(individuals, individual_masks)
        return self._fwd_with_combined(
            x,
            context,
            rope_emb,
            transformer_options,
            combined,
            combined_mask,
            mask,
            state["fusion_mode"],
            float(state["strength"]),
        )

    def _fwd_output_avg(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        rope_emb: torch.Tensor | None,
        transformer_options: dict[str, Any],
        individuals: list[torch.Tensor],
        individual_masks: list[torch.Tensor | None],
        weights: list[float],
        mask: list[bool],
        fusion_mode: str,
        strength: float,
    ) -> torch.Tensor:
        normalized_weights = _normalize_weights(weights) if self._state.get("normalize_weights", True) else list(weights)
        force_collect = self._state.get("artist_static_capture", False) and fusion_mode != FUSION_CONCAT_WITH_BASE

        if self._deferred_cache_enabled(transformer_options):
            deferred = self._fwd_output_avg_deferred(
                x,
                context,
                rope_emb,
                transformer_options,
                individuals,
                individual_masks,
                normalized_weights,
                mask,
                fusion_mode,
                strength,
            )
            if deferred is not None:
                return deferred

        if force_collect:
            artist_outputs = self._artist_outputs_with_static_cache(
                x,
                context,
                rope_emb,
                transformer_options,
                individuals,
                individual_masks,
                fusion_mode,
            )
            artist_total = self._weighted_sum_outputs(artist_outputs, normalized_weights)
        else:
            artist_total = self._output_avg_artist_total(
                x,
                context,
                rope_emb,
                transformer_options,
                individuals,
                individual_masks,
                normalized_weights,
                fusion_mode,
            )
        artist_total = self._apply_ema(artist_total, fusion_mode)

        if fusion_mode == FUSION_INTERPOLATE and strength == 1.0 and all(mask):
            return artist_total

        base_out = self.original(
            x,
            context,
            rope_emb=rope_emb,
            transformer_options=transformer_options,
        )
        return self._apply_fusion(base_out, artist_total, mask, fusion_mode, strength)

    def _output_avg_artist_total(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        rope_emb: torch.Tensor | None,
        transformer_options: dict[str, Any],
        individuals: list[torch.Tensor],
        individual_masks: list[torch.Tensor | None],
        normalized_weights: list[float],
        fusion_mode: str,
    ) -> torch.Tensor:
        artist_outputs = self._collect_artist_outputs(
            x,
            context,
            rope_emb,
            transformer_options,
            individuals,
            individual_masks,
            fusion_mode,
        )
        return self._weighted_sum_outputs(artist_outputs, normalized_weights)

    def _collect_artist_outputs(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        rope_emb: torch.Tensor | None,
        transformer_options: dict[str, Any],
        individuals: list[torch.Tensor],
        individual_masks: list[torch.Tensor | None],
        fusion_mode: str,
    ) -> list[torch.Tensor]:
        if len(individuals) >= 2 and not self._state.get("_disable_batched", False):
            try:
                return self._batched_artists_outputs_only(
                    x,
                    context,
                    rope_emb,
                    transformer_options,
                    individuals,
                    individual_masks,
                    fusion_mode,
                )
            except Exception as exc:
                if not self._state.get("_warned_batched", False):
                    logger.warning(
                        "[AnimaArtistMixer] batched artist outputs failed; using serial fallback: %s",
                        exc,
                    )
                    self._state["_warned_batched"] = True
                    self._state["_disable_batched"] = True

        batch_size = context.shape[0]
        outputs: list[torch.Tensor] = []
        for artist, artist_mask in zip(individuals, individual_masks):
            artist_b = _broadcast_batch(artist, batch_size).to(device=context.device, dtype=context.dtype)
            artist_mask_b = _fit_mask_to_context(artist_mask, artist_b)

            if fusion_mode == FUSION_CONCAT_WITH_BASE:
                base_mask = _mask_or_ones(_mask_from_options(transformer_options, context), context)
                kv = _cat_context(context, artist_b)
                kv_mask = _cat_context(base_mask, _mask_or_ones(artist_mask_b, artist_b))
            else:
                kv = artist_b
                kv_mask = artist_mask_b

            outputs.append(
                self._call_original(
                    x,
                    kv,
                    rope_emb,
                    _with_negpip_mask(transformer_options, kv_mask),
                    forge_couple_artist=True,
                )
            )

        if not outputs:
            raise RuntimeError("artist output list was empty")
        return outputs

    def _weighted_sum_outputs(
        self,
        outputs: list[torch.Tensor],
        weights: list[float],
    ) -> torch.Tensor:
        result = None
        for output, weight in zip(outputs, weights):
            result = output * weight if result is None else result + output * weight
        if result is None:
            raise RuntimeError("artist output list was empty")
        return result

    def _static_cache_signature(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        transformer_options: dict[str, Any],
        individuals: list[torch.Tensor],
        individual_masks: list[torch.Tensor | None],
        fusion_mode: str,
    ) -> tuple[Any, ...]:
        cond_or_uncond = transformer_options.get("cond_or_uncond") if isinstance(transformer_options, dict) else None
        negpip_mask = transformer_options.get("negpip_mask") if isinstance(transformer_options, dict) else None
        return (
            tuple(x.shape),
            tuple(context.shape),
            str(x.dtype),
            str(context.dtype),
            tuple(cond_or_uncond or ()),
            fusion_mode,
            tuple(tuple(artist.shape) for artist in individuals),
            tuple(_shape_tuple(mask) for mask in individual_masks),
            _shape_tuple(negpip_mask),
        )

    def _reset_static_cache(self) -> None:
        self._static_signature = None
        self._static_seen_sigmas.clear()
        self._static_accumulator = None
        self._static_count = 0
        self._static_frozen_outputs = None
        self._static_max_sigma = None

    def _maybe_reset_static_cache(
        self,
        signature: tuple[Any, ...],
        current_sigma: float | None,
    ) -> None:
        if self._static_signature is not None and self._static_signature != signature:
            self._reset_static_cache()

        if current_sigma is not None:
            if self._static_max_sigma is not None and float(current_sigma) > float(self._static_max_sigma) + 1e-3:
                self._reset_static_cache()
            self._static_max_sigma = (
                float(current_sigma)
                if self._static_max_sigma is None
                else max(float(self._static_max_sigma), float(current_sigma))
            )

        if self._static_signature is None:
            self._static_signature = signature

    def _artist_outputs_with_static_cache(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        rope_emb: torch.Tensor | None,
        transformer_options: dict[str, Any],
        individuals: list[torch.Tensor],
        individual_masks: list[torch.Tensor | None],
        fusion_mode: str,
    ) -> list[torch.Tensor]:
        capture_k = max(1, min(STATIC_CAPTURE_K_MAX, int(self._state.get("static_capture_k", STATIC_CAPTURE_K_DEFAULT))))
        current_sigma = self._state.get("current_sigma")
        current_sigma = float(current_sigma) if current_sigma is not None else None
        signature = self._static_cache_signature(
            x,
            context,
            transformer_options,
            individuals,
            individual_masks,
            fusion_mode,
        )
        self._maybe_reset_static_cache(signature, current_sigma)

        if self._static_frozen_outputs is not None:
            return [
                output.to(device=context.device, dtype=x.dtype)
                for output in self._static_frozen_outputs
            ]

        outputs = self._collect_artist_outputs(
            x,
            context,
            rope_emb,
            transformer_options,
            individuals,
            individual_masks,
            fusion_mode,
        )

        sigma_key = round(current_sigma, 6) if current_sigma is not None else float(self._static_count)
        if sigma_key not in self._static_seen_sigmas:
            self._static_seen_sigmas.add(sigma_key)
            detached = [output.detach() for output in outputs]
            if self._static_accumulator is None:
                self._static_accumulator = [output.clone() for output in detached]
            else:
                self._static_accumulator = [
                    acc.to(device=output.device, dtype=output.dtype) + output
                    for acc, output in zip(self._static_accumulator, detached)
                ]
            self._static_count += 1

            if self._static_count >= capture_k and self._static_accumulator is not None:
                self._static_frozen_outputs = [
                    (acc / float(self._static_count)).detach()
                    for acc in self._static_accumulator
                ]
                return [
                    output.to(device=context.device, dtype=x.dtype)
                    for output in self._static_frozen_outputs
                ]

        return outputs

    def _maybe_reset_ema(self) -> None:
        current_sigma = self._state.get("current_sigma")
        if current_sigma is None:
            return
        previous_sigma = self._state.get("_ema_last_sigma")
        if previous_sigma is None or float(current_sigma) > float(previous_sigma) + 1e-3:
            self._state["_ema_cache"] = {}
        self._state["_ema_last_sigma"] = float(current_sigma)

    def _apply_ema(self, artist_total: torch.Tensor, fusion_mode: str) -> torch.Tensor:
        if fusion_mode not in (FUSION_INTERPOLATE, FUSION_BASE_PRESERVE):
            return artist_total

        ema_alpha = float(self._state.get("artist_ema_alpha", 0.0))
        if ema_alpha <= 0.0:
            return artist_total

        self._maybe_reset_ema()
        cache = self._state.setdefault("_ema_cache", {})
        previous = cache.get(self._layer_idx)
        if previous is not None and previous.shape == artist_total.shape:
            artist_total = ema_alpha * previous.to(artist_total) + (1.0 - ema_alpha) * artist_total
        cache[self._layer_idx] = artist_total.detach()
        return artist_total

    def _apply_fusion(
        self,
        base_out: torch.Tensor,
        artist_total: torch.Tensor,
        mask: list[bool],
        fusion_mode: str,
        strength: float,
    ) -> torch.Tensor:
        if fusion_mode == FUSION_BASE_PRESERVE:
            delta = _project_perpendicular(artist_total - base_out, base_out)
            out = base_out.clone()
            for row, hit in enumerate(mask):
                if hit:
                    out[row] = base_out[row] + delta[row] * strength
            return out

        out = base_out.clone()
        for row, hit in enumerate(mask):
            if hit:
                out[row] = base_out[row] * (1.0 - strength) + artist_total[row] * strength
        return out

    def _deferred_cache_enabled(self, transformer_options: dict[str, Any]) -> bool:
        state = self._state
        if not state.get("deferred_cache", False):
            return False
        if state.get("_deferred_disabled_forge_couple", False):
            return False
        if state.get("combine_mode") != COMBINE_OUTPUT_AVG:
            return False
        if state.get("artist_static_capture", False):
            return False
        if state.get("deferred_warmup_sigma") is None:
            return False

        if _extract_forge_couple_state(self.original) is not None:
            state["_deferred_disabled_forge_couple"] = True
            if not state.get("_warned_deferred_forge_couple", False):
                logger.info("[AnimaArtistMixer] Deferred Cache disabled because Forge Couple was detected.")
                state["_warned_deferred_forge_couple"] = True
            return False

        return True

    def _deferred_cache_signature(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        base_out: torch.Tensor,
        transformer_options: dict[str, Any],
        mask: list[bool],
        fusion_mode: str,
        strength: float,
    ) -> tuple[Any, ...]:
        negpip_mask = transformer_options.get("negpip_mask") if isinstance(transformer_options, dict) else None
        cond_or_uncond = transformer_options.get("cond_or_uncond") if isinstance(transformer_options, dict) else None
        return (
            tuple(x.shape),
            tuple(base_out.shape),
            tuple(context.shape),
            tuple(cond_or_uncond or ()),
            tuple(mask),
            fusion_mode,
            round(float(strength), 6),
            _shape_tuple(negpip_mask),
        )

    def _reset_deferred_cache(self) -> None:
        self._deferred_signature = None
        self._deferred_delta_sum = None
        self._deferred_count = 0

    def _fwd_output_avg_deferred(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        rope_emb: torch.Tensor | None,
        transformer_options: dict[str, Any],
        individuals: list[torch.Tensor],
        individual_masks: list[torch.Tensor | None],
        normalized_weights: list[float],
        mask: list[bool],
        fusion_mode: str,
        strength: float,
    ) -> torch.Tensor | None:
        current_sigma = self._state.get("current_sigma")
        warmup_sigma = self._state.get("deferred_warmup_sigma")
        if current_sigma is None or warmup_sigma is None:
            return None

        base_out = self.original(
            x,
            context,
            rope_emb=rope_emb,
            transformer_options=transformer_options,
        )
        signature = self._deferred_cache_signature(
            x,
            context,
            base_out,
            transformer_options,
            mask,
            fusion_mode,
            strength,
        )
        if self._deferred_signature is not None and self._deferred_signature != signature:
            self._reset_deferred_cache()

        in_warmup = float(current_sigma) >= float(warmup_sigma)
        if not in_warmup and self._deferred_delta_sum is not None and self._deferred_count > 0:
            cached_delta = (self._deferred_delta_sum / float(self._deferred_count)).to(
                device=base_out.device,
                dtype=base_out.dtype,
            )
            cached_artist_total = self._apply_ema(base_out + cached_delta, fusion_mode)
            return self._apply_fusion(base_out, cached_artist_total, mask, fusion_mode, strength)

        artist_total = self._output_avg_artist_total(
            x,
            context,
            rope_emb,
            transformer_options,
            individuals,
            individual_masks,
            normalized_weights,
            fusion_mode,
        )
        artist_total = self._apply_ema(artist_total, fusion_mode)

        if in_warmup:
            delta = (artist_total - base_out).detach()
            if self._deferred_signature != signature:
                self._deferred_signature = signature
                self._deferred_delta_sum = delta.clone()
                self._deferred_count = 1
            elif self._deferred_delta_sum is not None:
                self._deferred_delta_sum = self._deferred_delta_sum + delta
                self._deferred_count += 1

        return self._apply_fusion(base_out, artist_total, mask, fusion_mode, strength)

    def _truncate_delta_lowrank(self, delta: torch.Tensor, lowrank_k: int) -> torch.Tensor:
        if delta.dim() < 2:
            return delta

        rows = delta.shape[-2]
        cols = delta.shape[-1]
        rank = max(1, min(int(lowrank_k), rows, cols))
        if rank >= min(rows, cols):
            return delta

        original_shape = delta.shape
        matrix = delta.float().reshape(-1, rows, cols)
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        truncated = (u[:, :, :rank] * s[:, :rank].unsqueeze(-2)) @ vh[:, :rank, :]
        return truncated.reshape(original_shape).to(device=delta.device, dtype=delta.dtype)

    def _fwd_lowrank_avg(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        rope_emb: torch.Tensor | None,
        transformer_options: dict[str, Any],
        individuals: list[torch.Tensor],
        individual_masks: list[torch.Tensor | None],
        weights: list[float],
        mask: list[bool],
        fusion_mode: str,
        strength: float,
    ) -> torch.Tensor:
        lowrank_k = max(1, int(self._state.get("lowrank_k", 1)))
        if lowrank_k >= len(individuals):
            return self._fwd_output_avg(
                x,
                context,
                rope_emb,
                transformer_options,
                individuals,
                individual_masks,
                weights,
                mask,
                fusion_mode,
                strength,
            )

        normalized_weights = _normalize_weights(weights) if self._state.get("normalize_weights", True) else list(weights)
        base_out = self.original(
            x,
            context,
            rope_emb=rope_emb,
            transformer_options=transformer_options,
        )

        try:
            if self._state.get("artist_static_capture", False) and fusion_mode != FUSION_CONCAT_WITH_BASE:
                artist_outputs = self._artist_outputs_with_static_cache(
                    x,
                    context,
                    rope_emb,
                    transformer_options,
                    individuals,
                    individual_masks,
                    fusion_mode,
                )
            else:
                artist_outputs = self._collect_artist_outputs(
                    x,
                    context,
                    rope_emb,
                    transformer_options,
                    individuals,
                    individual_masks,
                    fusion_mode,
                )

            delta_total = torch.zeros_like(base_out)
            for artist_out, weight in zip(artist_outputs, normalized_weights):
                delta_total = delta_total + self._truncate_delta_lowrank(artist_out - base_out, lowrank_k) * weight
            artist_total = base_out + delta_total
        except Exception as exc:
            if not self._state.get("_warned_lowrank", False):
                logger.warning(
                    "[AnimaArtistMixer] lowrank_avg failed; using output_avg fallback: %s",
                    exc,
                )
                self._state["_warned_lowrank"] = True
            return self._fwd_output_avg(
                x,
                context,
                rope_emb,
                transformer_options,
                individuals,
                individual_masks,
                weights,
                mask,
                fusion_mode,
                strength,
            )

        artist_total = self._apply_ema(artist_total, fusion_mode)
        return self._apply_fusion(base_out, artist_total, mask, fusion_mode, strength)

    def _batched_artists_outputs_only(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        rope_emb: torch.Tensor | None,
        transformer_options: dict[str, Any],
        individuals: list[torch.Tensor],
        individual_masks: list[torch.Tensor | None],
        fusion_mode: str,
    ) -> list[torch.Tensor]:
        batch_size = context.shape[0]
        artist_count = len(individuals)
        kv_list = []
        kv_mask_list = []
        base_mask = _mask_from_options(transformer_options, context)

        for artist, artist_mask in zip(individuals, individual_masks):
            artist_b = _broadcast_batch(artist, batch_size).to(device=context.device, dtype=context.dtype)
            artist_mask_b = _fit_mask_to_context(artist_mask, artist_b)

            if fusion_mode == FUSION_CONCAT_WITH_BASE:
                kv_list.append(_cat_context(context, artist_b))
                kv_mask_list.append(_cat_context(_mask_or_ones(base_mask, context), _mask_or_ones(artist_mask_b, artist_b)))
            else:
                kv_list.append(artist_b)
                kv_mask_list.append(artist_mask_b)

        kv_shapes = {tuple(kv.shape[1:]) for kv in kv_list}
        if len(kv_shapes) > 1:
            raise ValueError(f"incompatible artist K/V shapes: {kv_shapes}")

        x_rep = x.repeat(artist_count, *([1] * (x.dim() - 1)))
        kv_stacked = torch.cat(kv_list, dim=0)

        rope_rep = rope_emb
        if torch.is_tensor(rope_emb) and rope_emb.dim() > 0 and rope_emb.shape[0] == batch_size:
            rope_rep = rope_emb.repeat(artist_count, *([1] * (rope_emb.dim() - 1)))

        next_options = dict(transformer_options)
        if any(mask is not None for mask in kv_mask_list):
            next_options["negpip_mask"] = torch.cat(
                [
                    mask if mask is not None else _ones_mask_like(kv)
                    for kv, mask in zip(kv_list, kv_mask_list)
                ],
                dim=0,
            )
        else:
            next_options.pop("negpip_mask", None)

        cond_or_uncond = next_options.get("cond_or_uncond")
        if cond_or_uncond is not None:
            next_options["cond_or_uncond"] = list(cond_or_uncond) * artist_count

        out = self._call_original(
            x_rep,
            kv_stacked,
            rope_rep,
            next_options,
            forge_couple_artist=True,
        )
        out = out.view(artist_count, batch_size, *out.shape[1:])
        return [chunk for chunk in out.unbind(dim=0)]

    def _batched_artists_forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        rope_emb: torch.Tensor | None,
        transformer_options: dict[str, Any],
        individuals: list[torch.Tensor],
        individual_masks: list[torch.Tensor | None],
        weights: list[float],
        fusion_mode: str,
    ) -> torch.Tensor:
        out = torch.stack(
            self._batched_artists_outputs_only(
                x,
                context,
                rope_emb,
                transformer_options,
                individuals,
                individual_masks,
                fusion_mode,
            ),
            dim=0,
        )
        weight_tensor = torch.tensor(weights, device=out.device, dtype=out.dtype).view(
            out.shape[0],
            *([1] * (out.dim() - 1)),
        )
        return (out * weight_tensor).sum(dim=0)

    def _fwd_with_combined(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        rope_emb: torch.Tensor | None,
        transformer_options: dict[str, Any],
        combined: torch.Tensor,
        combined_mask: torch.Tensor | None,
        mask: list[bool],
        fusion_mode: str,
        strength: float,
    ) -> torch.Tensor:
        batch_size = context.shape[0]
        artist_b = _broadcast_batch(combined, batch_size).to(device=context.device, dtype=context.dtype)
        artist_mask_b = _fit_mask_to_context(combined_mask, artist_b)

        if fusion_mode in (FUSION_INTERPOLATE, FUSION_BASE_PRESERVE):
            base_out = self.original(
                x,
                context,
                rope_emb=rope_emb,
                transformer_options=transformer_options,
            )
            artist_out = self._call_original(
                x,
                artist_b,
                rope_emb,
                _with_negpip_mask(transformer_options, artist_mask_b),
                forge_couple_artist=True,
            )
            artist_out = self._apply_ema(artist_out, fusion_mode)
            return self._apply_fusion(base_out, artist_out, mask, fusion_mode, strength)

        extension = torch.zeros_like(artist_b)
        extension_mask = _ones_mask_like(artist_b)
        if artist_mask_b is not None:
            extension_mask = torch.ones_like(artist_mask_b)

        for row, hit in enumerate(mask):
            if hit:
                extension[row] = artist_b[row]
                if artist_mask_b is not None:
                    extension_mask[row] = artist_mask_b[row]
        merged = _cat_context(context, extension)
        base_mask = _mask_or_ones(_mask_from_options(transformer_options, context), context)
        merged_mask = _cat_context(base_mask, extension_mask)
        return self._call_original(
            x,
            merged,
            rope_emb,
            _with_negpip_mask(transformer_options, merged_mask),
            forge_couple_artist=True,
        )


def _make_runtime_cross_attn_wrapper(
    state: dict[str, Any],
    previous_wrapper: Any,
    patch_pairs: list[tuple[Any, nn.Module, _CrossAttnWrapper]],
):
    def wrapper(apply_model, options: dict[str, Any]):
        timestep = options.get("timestep")
        if timestep is not None:
            try:
                state["current_sigma"] = float(timestep.flatten()[0].item())
            except Exception:
                pass

        restore_pairs = []
        try:
            for block, _original, patched_cross_attn in patch_pairs:
                current_cross_attn = getattr(block, "cross_attn")
                restore_pairs.append((block, current_cross_attn))
                if current_cross_attn is not patched_cross_attn:
                    setattr(block, "cross_attn", patched_cross_attn)

            if previous_wrapper is not None:
                return previous_wrapper(apply_model, options)

            return apply_model(options["input"], options["timestep"], **options["c"])
        finally:
            for block, previous_cross_attn in reversed(restore_pairs):
                setattr(block, "cross_attn", previous_cross_attn)

    return wrapper


def _is_anima_processing_model(sd_model: Any) -> bool:
    return sd_model.__class__.__name__ == "Anima"


def _current_base_prompts(p: Any, kwargs: dict[str, Any]) -> list[str]:
    c = kwargs.get("c")
    if c is not None and c is getattr(p, "hr_c", None):
        hr_prompts = getattr(p, "hr_prompts", None)
        if isinstance(hr_prompts, list) and hr_prompts:
            return [str(prompt or "") for prompt in hr_prompts]

    prompts = getattr(p, "prompts", None)
    if isinstance(prompts, list) and prompts:
        return [str(prompt or "") for prompt in prompts]

    return [str(getattr(p, "prompt", "") or "")]


def _encode_artist_conditionings(
    sd_model: Any,
    artists: list[str],
    base_prompts: list[str],
) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
    expected_batch = len(base_prompts)
    individuals: list[torch.Tensor] = []
    individual_masks: list[torch.Tensor | None] = []

    with torch.inference_mode():
        for artist in artists:
            texts = [
                f"{artist}\n{base.strip()}" if base.strip() else artist
                for base in base_prompts
            ]
            conditioning = sd_model.get_learned_conditioning(texts)
            tensor = _conditioning_to_batch_tensor(conditioning, expected_batch)
            if tensor is None:
                raise ValueError(f"artist conditioning is empty: {artist!r}")

            mask = _conditioning_to_batch_tensor(
                conditioning,
                expected_batch,
                ("c_negpip_mask", "negpip_mask"),
            )
            mask = _fit_mask_to_context(mask, tensor) if mask is not None else None
            individuals.append(tensor)
            individual_masks.append(mask)

    return individuals, individual_masks


def _artist_summary_vector(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach().float()
    if value.dim() < 2:
        return value.flatten()
    value = value.reshape(-1, value.shape[-1])
    return value.mean(dim=0)


def _similar_artist_clusters(
    individuals: list[torch.Tensor],
    threshold: float,
) -> list[list[int]]:
    count = len(individuals)
    if count <= 1:
        return [[index] for index in range(count)]

    summaries = torch.stack([_artist_summary_vector(tensor) for tensor in individuals], dim=0)
    summaries = torch.nn.functional.normalize(summaries, dim=-1)
    similarities = summaries @ summaries.T

    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(count):
        for right in range(left + 1, count):
            if float(similarities[left, right].item()) >= threshold:
                union(left, right)

    grouped: dict[int, list[int]] = {}
    for index in range(count):
        grouped.setdefault(find(index), []).append(index)

    return sorted(grouped.values(), key=lambda members: members[0])


def _has_nontrivial_negpip_mask(mask: torch.Tensor | None) -> bool:
    if mask is None:
        return False
    try:
        return bool(torch.any(torch.abs(mask.detach().float() - 1.0) > 1e-6).item())
    except Exception:
        return True


def _merge_similar_artists(
    artists: list[str],
    individuals: list[torch.Tensor],
    individual_masks: list[torch.Tensor | None],
    base_weights: list[float],
    threshold: float,
) -> tuple[list[torch.Tensor], list[torch.Tensor | None], list[float], str]:
    if len(base_weights) < len(individuals):
        base_weights = list(base_weights) + [1.0] * (len(individuals) - len(base_weights))

    if len(individuals) <= 1:
        return individuals, individual_masks, list(base_weights), ""

    threshold = _clamp01(threshold)
    clusters = _similar_artist_clusters(individuals, threshold)

    merged_individuals: list[torch.Tensor] = []
    merged_masks: list[torch.Tensor | None] = []
    merged_weights: list[float] = []
    summary_parts: list[str] = []

    for members in clusters:
        member_shapes = {tuple(individuals[index].shape) for index in members}
        if (
            len(members) == 1
            or len(member_shapes) > 1
            or any(_has_nontrivial_negpip_mask(individual_masks[index]) for index in members)
        ):
            for index in members:
                merged_individuals.append(individuals[index])
                merged_masks.append(individual_masks[index])
                merged_weights.append(float(base_weights[index]))
            continue

        cluster_weight = sum(float(base_weights[index]) for index in members)
        if cluster_weight > 1e-8:
            merged = None
            for index in members:
                part = individuals[index] * (float(base_weights[index]) / cluster_weight)
                merged = part if merged is None else merged + part
            if merged is None:
                stacked = torch.stack([individuals[index] for index in members], dim=0)
                merged = stacked.mean(dim=0)
        else:
            stacked = torch.stack([individuals[index] for index in members], dim=0)
            merged = stacked.mean(dim=0)

        merged_individuals.append(merged)
        merged_masks.append(None)
        merged_weights.append(float(cluster_weight))
        labels = ", ".join(artists[index] for index in members)
        summary_parts.append(f"[{labels}] -> weight {cluster_weight:.3g}")

    return merged_individuals, merged_masks, merged_weights, "; ".join(summary_parts)


def _target_blocks(
    num_blocks: int,
    start_block: int,
    end_block: int,
    layer_filter: str,
) -> list[int]:
    explicit_blocks = _parse_layer_filter(layer_filter, num_blocks)
    if explicit_blocks is not None:
        return explicit_blocks

    start = max(0, int(start_block))
    end = num_blocks - 1 if int(end_block) < 0 else min(num_blocks - 1, int(end_block))
    if start > end:
        raise ValueError(f"start_block={start} > end_block={end} (num_blocks={num_blocks})")

    return list(range(start, end + 1))


def _sigma_range(unet: Any, start_percent: float, end_percent: float) -> tuple[float, float] | None:
    if start_percent <= 0.0 and end_percent >= 1.0:
        return None

    predictor = getattr(getattr(unet, "model", None), "predictor", None)
    if predictor is None or not hasattr(predictor, "percent_to_sigma"):
        logger.warning("[AnimaArtistMixer] Cannot resolve sigma range; percent gating is disabled.")
        return None

    start_sigma = float(predictor.percent_to_sigma(float(start_percent)))
    end_sigma = float(predictor.percent_to_sigma(float(end_percent)))
    lo, hi = sorted([end_sigma, start_sigma])
    return lo, hi


def _sigma_at_percent(unet: Any, percent: float) -> float | None:
    predictor = getattr(getattr(unet, "model", None), "predictor", None)
    if predictor is None or not hasattr(predictor, "percent_to_sigma"):
        logger.warning("[AnimaArtistMixer] Cannot resolve deferred cache warmup sigma; Deferred Cache is disabled.")
        return None

    return float(predictor.percent_to_sigma(float(percent)))


def _diffusion_model_from_unet(unet: Any) -> Any:
    try:
        return unet.get_model_object("diffusion_model")
    except Exception:
        return unet.model.diffusion_model


def _unet_has_forge_couple(unet: Any) -> bool:
    try:
        diffusion_model = _diffusion_model_from_unet(unet)
    except Exception:
        return False

    blocks = getattr(diffusion_model, "blocks", None)
    if not blocks:
        return False

    for block in blocks:
        cross_attn = getattr(block, "cross_attn", None)
        if cross_attn is None:
            continue
        inner = _unwrap_cross_attn(cross_attn)
        if _extract_forge_couple_state(inner) is not None:
            return True

    return False


def _patch_unet(
    unet: Any,
    individuals: list[torch.Tensor],
    individual_masks: list[torch.Tensor | None],
    user_weights: list[float],
    combine_mode: str,
    fusion_mode: str,
    strength: float,
    apply_to_uncond: bool,
    normalize_weights: bool,
    start_block: int,
    end_block: int,
    start_percent: float,
    end_percent: float,
    layer_filter: str,
    deferred_cache: bool,
    deferred_warmup_sigma: float | None,
    lowrank_k: int,
    artist_ema_alpha: float,
    artist_static_capture: bool,
    static_capture_k: int,
    has_explicit_weights: bool,
) -> Any:
    diffusion_model = _diffusion_model_from_unet(unet)

    ok, num_blocks, message = _validate_diffusion_model(diffusion_model)
    if not ok:
        raise ValueError(message)

    target_blocks = _target_blocks(num_blocks, start_block, end_block, layer_filter)
    sigma_range = _sigma_range(unet, start_percent, end_percent)

    patched_unet = unet.clone()
    state = {
        "fusion_mode": fusion_mode,
        "combine_mode": combine_mode,
        "strength": float(strength),
        "apply_to_uncond": bool(apply_to_uncond),
        "normalize_weights": bool(normalize_weights),
        "individuals": individuals,
        "individual_masks": individual_masks,
        "user_weights": user_weights,
        "sigma_range": sigma_range,
        "current_sigma": None,
        "deferred_cache": bool(deferred_cache and combine_mode == COMBINE_OUTPUT_AVG and deferred_warmup_sigma is not None),
        "deferred_warmup_sigma": deferred_warmup_sigma,
        "lowrank_k": max(1, int(lowrank_k)),
        "artist_ema_alpha": _clamp01(artist_ema_alpha),
        "artist_static_capture": bool(artist_static_capture),
        "static_capture_k": max(1, min(STATIC_CAPTURE_K_MAX, int(static_capture_k))),
        "has_explicit_weights": bool(has_explicit_weights),
        "_ema_cache": {},
        "_ema_last_sigma": None,
    }

    patch_pairs = []

    for block_index in target_blocks:
        block = diffusion_model.blocks[block_index]
        inner = _unwrap_cross_attn(block.cross_attn)
        patch_pairs.append((block, inner, _CrossAttnWrapper(inner, state, block_index)))

    previous = patched_unet.model_options.get("model_function_wrapper")
    patched_unet.set_model_unet_function_wrapper(
        _make_runtime_cross_attn_wrapper(state, previous, patch_pairs)
    )

    return patched_unet


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _record_generation_params(
    p: Any,
    artists: list[str],
    effective_artist_count: int,
    combine_mode: str,
    fusion_mode: str,
    strength: float,
    apply_to_uncond: bool,
    normalize_weights: bool,
    start_block: int,
    end_block: int,
    start_percent: float,
    end_percent: float,
    layer_filter: str,
    deferred_cache: bool,
    deferred_warmup_percent: float,
    merge_similar: bool,
    merge_threshold: float,
    merge_summary: str,
    lowrank_k: int,
    artist_ema_alpha: float,
    artist_static_capture: bool,
    static_capture_k: int,
    has_explicit_weights: bool,
) -> None:
    p.extra_generation_params["AnimaArtistCrossAttn"] = f"Active ({len(artists)} artists)"
    p.extra_generation_params["Anima Artist Chain"] = ", ".join(artists)
    p.extra_generation_params["AnimaArtistCrossAttn Strength"] = float(strength)
    p.extra_generation_params["AnimaArtistCrossAttn Combine"] = combine_mode
    p.extra_generation_params["AnimaArtistCrossAttn Fusion"] = fusion_mode
    p.extra_generation_params["AnimaArtistCrossAttn Apply Uncond"] = bool(apply_to_uncond)
    p.extra_generation_params["AnimaArtistCrossAttn Normalize Weights"] = bool(normalize_weights)
    p.extra_generation_params["AnimaArtistCrossAttn Explicit Weights"] = bool(has_explicit_weights)
    p.extra_generation_params["AnimaArtistCrossAttn Start Block"] = int(start_block)
    p.extra_generation_params["AnimaArtistCrossAttn End Block"] = int(end_block)
    p.extra_generation_params["AnimaArtistCrossAttn Start Percent"] = float(start_percent)
    p.extra_generation_params["AnimaArtistCrossAttn End Percent"] = float(end_percent)
    if layer_filter:
        p.extra_generation_params["AnimaArtistCrossAttn Layer Filter"] = layer_filter
    p.extra_generation_params["AnimaArtistCrossAttn Deferred Cache"] = bool(deferred_cache)
    p.extra_generation_params["AnimaArtistCrossAttn Deferred Cache Warmup Percent"] = float(deferred_warmup_percent)
    p.extra_generation_params["AnimaArtistCrossAttn Merge Similar Artists"] = bool(merge_similar)
    p.extra_generation_params["AnimaArtistCrossAttn Merge Similar Threshold"] = float(merge_threshold)
    p.extra_generation_params["AnimaArtistCrossAttn Lowrank K"] = int(lowrank_k)
    p.extra_generation_params["AnimaArtistCrossAttn Artist EMA Alpha"] = float(artist_ema_alpha)
    p.extra_generation_params["AnimaArtistCrossAttn Artist Static Capture"] = bool(artist_static_capture)
    p.extra_generation_params["AnimaArtistCrossAttn Static Capture K"] = int(static_capture_k)
    if effective_artist_count != len(artists):
        p.extra_generation_params["AnimaArtistCrossAttn Effective Artists"] = int(effective_artist_count)
    if merge_summary:
        p.extra_generation_params["AnimaArtistCrossAttn Merge Summary"] = merge_summary


class ScriptArtistStrengthTop(scripts.Script):
    section = "cfg"
    create_group = False
    sorting_priority = -1000

    def __init__(self):
        self.strength: gr.Slider | None = None
        self._registered_target: str | None = None

    def title(self):
        return "AnimaArtistCrossAttn Artist Strength"

    def show(self, is_img2img):
        if not _artist_strength_in_cfg_row():
            return None

        id_part = "img2img" if is_img2img else "txt2img"
        target_elem_id = f"{id_part}_distilled_cfg_scale"
        if self._registered_target != target_elem_id:
            self.on_before_component(lambda _x: self._inject_strength_slider(id_part), elem_id=target_elem_id)
            self._registered_target = target_elem_id

        return scripts.AlwaysVisible

    def _inject_strength_slider(self, id_part: str) -> None:
        if self.strength is None:
            self.strength = _create_strength_slider(
                id_part,
                label="Artist Strength",
                elem_id=f"{id_part}_anima_artist_strength",
                scale=4,
            )

    def ui(self, is_img2img):
        id_part = "img2img" if is_img2img else "txt2img"
        if self.strength is None:
            self._inject_strength_slider(id_part)

        self.infotext_fields = [(self.strength, "AnimaArtistCrossAttn Strength")]
        self.paste_field_names = ["AnimaArtistCrossAttn Strength"]
        return [self.strength]

    def process_before_every_sampling(self, p, strength: float, **kwargs) -> None:
        p._aam_artist_strength = _clamp_strength(strength)


class Script(scripts.Script):
    sorting_priority = 18138

    def title(self):
        return TITLE

    def show(self, _is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        id_part = "img2img" if is_img2img else "txt2img"
        strength_in_cfg_row = _artist_strength_in_cfg_row()
        artist_chain = _artist_chain_component(id_part)
        if artist_chain is None:
            logger.warning(
                "[AnimaArtistMixer] visible %s Artist Chain textbox was not registered; "
                "using hidden fallback for this UI session. Reload UI after startup.",
                id_part,
            )
            artist_chain = gr.Textbox(
                label="Artist Chain",
                elem_id=f"{id_part}_anima_artist_chain_hidden",
                visible=False,
            )

        with gr.Accordion(label=TITLE, open=False, elem_id=f"{id_part}_anima_artist_crossattn"):
            with gr.Row():
                combine_mode = gr.Radio(
                    choices=COMBINE_CHOICES,
                    value=COMBINE_OUTPUT_AVG,
                    label="Combine Mode",
                )
                fusion_mode = gr.Radio(
                    choices=FUSION_CHOICES,
                    value=FUSION_INTERPOLATE,
                    label="Fusion Mode",
                )
                if strength_in_cfg_row:
                    strength = gr.Number(
                        value=1.0,
                        label="Strength",
                        visible=False,
                        elem_id=f"{id_part}_anima_artist_strength_hidden",
                    )
                    strength.do_not_save_to_config = True
                else:
                    strength = _create_strength_slider(id_part, label="Strength")

            with gr.Row():
                apply_to_uncond = gr.Checkbox(value=False, label="Apply to Uncond")
                normalize_weights = gr.Checkbox(value=True, label="Normalize Weights")

            with gr.Accordion("Advanced", open=False):
                with gr.Row():
                    start_block = gr.Slider(
                        minimum=0,
                        maximum=63,
                        value=0,
                        step=1,
                        label="Start Block",
                    )
                    end_block = gr.Slider(
                        minimum=-1,
                        maximum=63,
                        value=-1,
                        step=1,
                        label="End Block",
                    )
                with gr.Row():
                    start_percent = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.0,
                        step=0.001,
                        label="Start Percent",
                    )
                    end_percent = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=1.0,
                        step=0.001,
                        label="End Percent",
                    )
                layer_filter = gr.Textbox(
                    value="",
                    label="Layer Filter",
                    lines=1,
                    max_lines=1,
                    placeholder="0,3,5-10,-1",
                )
                with gr.Row():
                    lowrank_k = gr.Slider(
                        minimum=1,
                        maximum=MAX_ARTISTS,
                        value=1,
                        step=1,
                        label="Lowrank K",
                    )
                    artist_ema_alpha = gr.Slider(
                        minimum=0.0,
                        maximum=0.95,
                        value=0.0,
                        step=0.05,
                        label="Artist EMA Alpha",
                    )
                with gr.Row():
                    artist_static_capture = gr.Checkbox(value=False, label="Experimental: Artist Static Capture")
                    static_capture_k = gr.Slider(
                        minimum=1,
                        maximum=STATIC_CAPTURE_K_MAX,
                        value=STATIC_CAPTURE_K_DEFAULT,
                        step=1,
                        label="Experimental: Static Capture K",
                    )
                with gr.Row():
                    deferred_cache = gr.Checkbox(value=False, label="Experimental: Deferred Cache")
                    deferred_warmup_percent = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.30,
                        step=0.01,
                        label="Experimental: Deferred Cache Warmup Percent",
                    )
                with gr.Row():
                    merge_similar = gr.Checkbox(value=False, label="Experimental: Merge Similar Artists")
                    merge_threshold = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.95,
                        step=0.01,
                        label="Experimental: Merge Similar Threshold",
                    )

        self.infotext_fields = [
            (artist_chain, "Anima Artist Chain"),
            (combine_mode, "AnimaArtistCrossAttn Combine"),
            (fusion_mode, "AnimaArtistCrossAttn Fusion"),
            (apply_to_uncond, "AnimaArtistCrossAttn Apply Uncond"),
            (normalize_weights, "AnimaArtistCrossAttn Normalize Weights"),
            (start_block, "AnimaArtistCrossAttn Start Block"),
            (end_block, "AnimaArtistCrossAttn End Block"),
            (start_percent, "AnimaArtistCrossAttn Start Percent"),
            (end_percent, "AnimaArtistCrossAttn End Percent"),
            (layer_filter, "AnimaArtistCrossAttn Layer Filter"),
            (deferred_cache, "AnimaArtistCrossAttn Deferred Cache"),
            (deferred_warmup_percent, "AnimaArtistCrossAttn Deferred Cache Warmup Percent"),
            (merge_similar, "AnimaArtistCrossAttn Merge Similar Artists"),
            (merge_threshold, "AnimaArtistCrossAttn Merge Similar Threshold"),
            (lowrank_k, "AnimaArtistCrossAttn Lowrank K"),
            (artist_ema_alpha, "AnimaArtistCrossAttn Artist EMA Alpha"),
            (artist_static_capture, "AnimaArtistCrossAttn Artist Static Capture"),
            (static_capture_k, "AnimaArtistCrossAttn Static Capture K"),
        ]
        if not strength_in_cfg_row:
            self.infotext_fields.insert(1, (strength, "AnimaArtistCrossAttn Strength"))
        self.paste_field_names = [name for _, name in self.infotext_fields]

        return [
            artist_chain,
            combine_mode,
            fusion_mode,
            strength,
            apply_to_uncond,
            normalize_weights,
            start_block,
            end_block,
            start_percent,
            end_percent,
            layer_filter,
            deferred_cache,
            deferred_warmup_percent,
            merge_similar,
            merge_threshold,
            lowrank_k,
            artist_ema_alpha,
            artist_static_capture,
            static_capture_k,
        ]

    def process_before_every_sampling(
        self,
        p,
        artist_chain: str,
        combine_mode: str,
        fusion_mode: str,
        strength: float,
        apply_to_uncond: bool,
        normalize_weights: bool,
        start_block: int,
        end_block: int,
        start_percent: float,
        end_percent: float,
        layer_filter: str,
        deferred_cache: bool,
        deferred_warmup_percent: float,
        merge_similar: bool,
        merge_threshold: float,
        lowrank_k: int,
        artist_ema_alpha: float,
        artist_static_capture: bool,
        static_capture_k: int,
        **kwargs,
    ) -> None:
        artist_entries = _split_artist_chain(artist_chain)
        artists, parsed_weights, has_explicit_weights = _parse_artist_weights(artist_entries)
        if not artists:
            return

        if not _is_anima_processing_model(p.sd_model):
            return

        if len(artists) > MAX_ARTISTS:
            logger.warning(
                "[AnimaArtistMixer] Artist count %d exceeds limit %d; truncating.",
                len(artists),
                MAX_ARTISTS,
            )
            artists = artists[:MAX_ARTISTS]
            artist_entries = artist_entries[:MAX_ARTISTS]
            parsed_weights = parsed_weights[:MAX_ARTISTS]

        if combine_mode not in COMBINE_CHOICES:
            combine_mode = COMBINE_OUTPUT_AVG
        if fusion_mode not in FUSION_CHOICES:
            fusion_mode = FUSION_INTERPOLATE

        if _artist_strength_in_cfg_row():
            strength = getattr(p, "_aam_artist_strength", strength)
        strength = _clamp_strength(strength)
        start_percent = _clamp01(start_percent)
        end_percent = _clamp01(end_percent)
        deferred_warmup_percent = _clamp01(deferred_warmup_percent)
        merge_threshold = _clamp01(merge_threshold)
        lowrank_k = max(1, min(MAX_ARTISTS, int(lowrank_k)))
        artist_ema_alpha = _clamp01(artist_ema_alpha)
        static_capture_k = max(1, min(STATIC_CAPTURE_K_MAX, int(static_capture_k)))

        unet = p.sd_model.forge_objects.unet
        forge_couple_detected = _unet_has_forge_couple(unet)
        output_avg_mode = combine_mode == COMBINE_OUTPUT_AVG
        static_capture_mode = combine_mode in (COMBINE_OUTPUT_AVG, COMBINE_LOWRANK_AVG)
        effective_static_capture = (
            bool(artist_static_capture)
            and static_capture_mode
            and fusion_mode != FUSION_CONCAT_WITH_BASE
            and not forge_couple_detected
        )
        effective_deferred_cache = bool(deferred_cache) and output_avg_mode and not forge_couple_detected and not effective_static_capture
        effective_merge_similar = bool(merge_similar) and output_avg_mode and not forge_couple_detected
        deferred_warmup_sigma = _sigma_at_percent(unet, deferred_warmup_percent) if effective_deferred_cache else None
        effective_deferred_cache = effective_deferred_cache and deferred_warmup_sigma is not None
        effective_normalize_weights = bool(normalize_weights) and not has_explicit_weights

        base_prompts = _current_base_prompts(p, kwargs)
        individuals, individual_masks = _encode_artist_conditionings(p.sd_model, artists, base_prompts)
        user_weights = list(parsed_weights)
        merge_summary = ""

        if effective_merge_similar:
            individuals, individual_masks, user_weights, merge_summary = _merge_similar_artists(
                artists,
                individuals,
                individual_masks,
                user_weights,
                merge_threshold,
            )

        patched_unet = _patch_unet(
            unet,
            individuals,
            individual_masks,
            user_weights,
            combine_mode,
            fusion_mode,
            strength,
            bool(apply_to_uncond),
            effective_normalize_weights,
            int(start_block),
            int(end_block),
            start_percent,
            end_percent,
            str(layer_filter or ""),
            effective_deferred_cache,
            deferred_warmup_sigma,
            lowrank_k,
            artist_ema_alpha,
            effective_static_capture,
            static_capture_k,
            has_explicit_weights,
        )
        p.sd_model.forge_objects.unet = patched_unet

        _record_generation_params(
            p,
            artist_entries,
            len(individuals),
            combine_mode,
            fusion_mode,
            strength,
            bool(apply_to_uncond),
            effective_normalize_weights,
            int(start_block),
            int(end_block),
            start_percent,
            end_percent,
            str(layer_filter or "").strip(),
            effective_deferred_cache,
            deferred_warmup_percent,
            effective_merge_similar,
            merge_threshold,
            merge_summary,
            lowrank_k,
            artist_ema_alpha,
            effective_static_capture,
            static_capture_k,
            has_explicit_weights,
        )
