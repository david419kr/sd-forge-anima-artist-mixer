from __future__ import annotations

import json
import logging
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import gradio as gr
import torch
import torch.nn as nn

import modules.scripts as scripts
from modules import script_callbacks, shared
from modules.ui_components import InputAccordion

logger = logging.getLogger(__name__)

TITLE = "AnimaArtistCrossAttn"
SECTION = ("anima_artist_mixer", "Anima Artist Mixer")

OPT_POSITION = "aam_artist_chain_position"
POSITION_ABOVE = "above"
POSITION_BETWEEN = "between"
POSITION_BELOW = "below"
POSITION_CHOICES = [POSITION_ABOVE, POSITION_BETWEEN, POSITION_BELOW]

SCRIPT_BASENAME = Path(__file__).name
UI_KEY_TXT = f"customscript/{SCRIPT_BASENAME}/txt2img/{TITLE}/value"
UI_KEY_IMG = f"customscript/{SCRIPT_BASENAME}/img2img/{TITLE}/value"

FUSION_INTERPOLATE = "interpolate"
FUSION_CONCAT_WITH_BASE = "concat_with_base"
FUSION_CHOICES = [FUSION_INTERPOLATE, FUSION_CONCAT_WITH_BASE]

COMBINE_CONCAT = "concat"
COMBINE_OUTPUT_AVG = "output_avg"
COMBINE_CHOICES = [COMBINE_OUTPUT_AVG, COMBINE_CONCAT]

MAX_ARTISTS = 32
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


script_callbacks.on_ui_settings(_register_settings)


def _artist_position() -> str:
    value = getattr(shared.opts, OPT_POSITION, POSITION_BETWEEN)
    return value if value in POSITION_CHOICES else POSITION_BETWEEN


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
    _ARTIST_COMPONENTS[id_part] = artist_chain
    return artist_chain


def _patch_toprow() -> None:
    from modules.ui_toprow import Toprow

    if getattr(Toprow.create_prompts, "_aam_patched", False):
        return

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
    create_prompts._aam_original = Toprow.create_prompts
    Toprow.create_prompts = create_prompts


script_callbacks.on_before_ui(_patch_toprow)


def _ui_config_path() -> Path:
    return Path(shared.data_path) / "ui-config.json"


def _read_ui_config() -> dict[str, Any]:
    path = _ui_config_path()
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("[AnimaArtistMixer] Failed to read ui-config.json: %s", exc)
        return {}


def _write_ui_config(config: dict[str, Any]) -> None:
    path = _ui_config_path()
    path.write_text(json.dumps(config, indent=4, ensure_ascii=False), encoding="utf-8")


def _startup_keys() -> tuple[str, str]:
    return UI_KEY_TXT, UI_KEY_IMG


def _read_startup_default(is_img2img: bool) -> bool:
    config = _read_ui_config()
    key = UI_KEY_IMG if is_img2img else UI_KEY_TXT
    if key in config:
        return bool(config[key])
    if UI_KEY_TXT in config:
        return bool(config[UI_KEY_TXT])
    if UI_KEY_IMG in config:
        return bool(config[UI_KEY_IMG])
    return False


def _startup_button_label(enabled: bool) -> str:
    return "Startup auto ON: ON" if enabled else "Startup auto ON: OFF"


def _toggle_startup_default_common() -> gr.update:
    config = _read_ui_config()
    keys = _startup_keys()
    current = bool(config.get(UI_KEY_TXT, False))
    new_value = not current
    for key in keys:
        config[key] = new_value
    _write_ui_config(config)
    return gr.update(value=_startup_button_label(new_value))


def _split_artist_chain(chain: str | None) -> list[str]:
    if not chain:
        return []

    normalized = str(chain).replace("，", ",").replace("\n", ",").replace("\r", ",")
    parts = [_EXTRA_NETWORK_RE.sub("", part).strip() for part in normalized.split(",")]
    return [part for part in parts if part]


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

    if context_3d.shape[-2] < 512:
        context_3d = torch.nn.functional.pad(context_3d, (0, 0, 0, 512 - context_3d.shape[-2]))
    elif context_3d.shape[-2] % 512 != 0:
        target_tokens = ((context_3d.shape[-2] + 511) // 512) * 512
        context_3d = torch.nn.functional.pad(context_3d, (0, 0, 0, target_tokens - context_3d.shape[-2]))

    context_chunks = context_3d.chunk(num_chunks, dim=0)

    negpip_mask = _mask_from_options(transformer_options, context_3d)
    negpip_chunks = negpip_mask.chunk(num_chunks, dim=0) if negpip_mask is not None else None

    new_x = []
    new_context = []
    new_negpip_masks = []

    for idx, cond_or_uncond in enumerate(cond_or_unconds):
        c_target = context_chunks[idx]
        if cond_or_uncond == 1:
            new_x.append(x_chunks[idx])
            new_context.append(c_target)
            if negpip_chunks is not None:
                new_negpip_masks.append(negpip_chunks[idx])
        else:
            new_x.append(x_chunks[idx].repeat(num_conds, *([1] * (x.dim() - 1))))
            new_context.append(c_target.repeat(num_conds, *([1] * (c_target.dim() - 1))))
            if negpip_chunks is not None:
                new_negpip_masks.append(negpip_chunks[idx].repeat(num_conds, *([1] * (negpip_chunks[idx].dim() - 1))))

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


class _CrossAttnWrapper(nn.Module):
    def __init__(self, original: nn.Module, shared_state: dict[str, Any], layer_idx: int):
        super().__init__()
        self.original = original
        self._state = shared_state
        self._layer_idx = layer_idx
        self._disabled = False

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
        if not state.get("enabled", False) or context is None:
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

        if state["combine_mode"] == COMBINE_OUTPUT_AVG:
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
        batch_size = context.shape[0]
        normalized_weights = _normalize_weights(weights) if self._state.get("normalize_weights", True) else list(weights)
        artist_total = None

        if len(individuals) >= 2 and not self._state.get("_disable_batched", False):
            try:
                artist_total = self._batched_artists_forward(
                    x,
                    context,
                    rope_emb,
                    transformer_options,
                    individuals,
                    individual_masks,
                    normalized_weights,
                    fusion_mode,
                )
            except Exception as exc:
                if not self._state.get("_warned_batched", False):
                    logger.warning(
                        "[AnimaArtistMixer] batched output_avg failed; using serial fallback: %s",
                        exc,
                    )
                    self._state["_warned_batched"] = True
                    self._state["_disable_batched"] = True
                artist_total = None

        if artist_total is None:
            for artist, artist_mask, weight in zip(individuals, individual_masks, normalized_weights):
                artist_b = _broadcast_batch(artist, batch_size).to(device=context.device, dtype=context.dtype)
                artist_mask_b = _fit_mask_to_context(artist_mask, artist_b)

                if fusion_mode == FUSION_CONCAT_WITH_BASE:
                    base_mask = _mask_or_ones(_mask_from_options(transformer_options, context), context)
                    kv = _cat_context(context, artist_b)
                    kv_mask = _cat_context(base_mask, _mask_or_ones(artist_mask_b, artist_b))
                else:
                    kv = artist_b
                    kv_mask = artist_mask_b

                out_i = self._call_original(
                    x,
                    kv,
                    rope_emb,
                    _with_negpip_mask(transformer_options, kv_mask),
                    forge_couple_artist=True,
                )
                artist_total = out_i * weight if artist_total is None else artist_total + out_i * weight

        if strength >= 1.0 and all(mask):
            return artist_total

        base_out = self.original(
            x,
            context,
            rope_emb=rope_emb,
            transformer_options=transformer_options,
        )
        out = base_out.clone()
        for row, hit in enumerate(mask):
            if hit:
                out[row] = base_out[row] * (1.0 - strength) + artist_total[row] * strength
        return out

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
        weight_tensor = torch.tensor(weights, device=out.device, dtype=out.dtype).view(
            artist_count,
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

        if fusion_mode == FUSION_INTERPOLATE:
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
            out = base_out.clone()
            for row, hit in enumerate(mask):
                if hit:
                    out[row] = base_out[row] * (1.0 - strength) + artist_out[row] * strength
            return out

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


def _patch_unet(
    unet: Any,
    individuals: list[torch.Tensor],
    individual_masks: list[torch.Tensor | None],
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
) -> Any:
    try:
        diffusion_model = unet.get_model_object("diffusion_model")
    except Exception:
        diffusion_model = unet.model.diffusion_model

    ok, num_blocks, message = _validate_diffusion_model(diffusion_model)
    if not ok:
        raise ValueError(message)

    target_blocks = _target_blocks(num_blocks, start_block, end_block, layer_filter)
    sigma_range = _sigma_range(unet, start_percent, end_percent)

    patched_unet = unet.clone()
    state = {
        "enabled": True,
        "fusion_mode": fusion_mode,
        "combine_mode": combine_mode,
        "strength": float(strength),
        "apply_to_uncond": bool(apply_to_uncond),
        "normalize_weights": bool(normalize_weights),
        "individuals": individuals,
        "individual_masks": individual_masks,
        "user_weights": [1.0] * len(individuals),
        "sigma_range": sigma_range,
        "current_sigma": None,
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
) -> None:
    p.extra_generation_params["Anima Artist Chain"] = ", ".join(artists)
    p.extra_generation_params["AnimaArtistCrossAttn Enabled"] = True
    p.extra_generation_params["AnimaArtistCrossAttn Strength"] = float(strength)
    p.extra_generation_params["AnimaArtistCrossAttn Combine"] = combine_mode
    p.extra_generation_params["AnimaArtistCrossAttn Fusion"] = fusion_mode
    p.extra_generation_params["AnimaArtistCrossAttn Apply Uncond"] = bool(apply_to_uncond)
    p.extra_generation_params["AnimaArtistCrossAttn Normalize Weights"] = bool(normalize_weights)
    p.extra_generation_params["AnimaArtistCrossAttn Start Block"] = int(start_block)
    p.extra_generation_params["AnimaArtistCrossAttn End Block"] = int(end_block)
    p.extra_generation_params["AnimaArtistCrossAttn Start Percent"] = float(start_percent)
    p.extra_generation_params["AnimaArtistCrossAttn End Percent"] = float(end_percent)
    if layer_filter:
        p.extra_generation_params["AnimaArtistCrossAttn Layer Filter"] = layer_filter


class Script(scripts.Script):
    sorting_priority = 18138

    def title(self):
        return TITLE

    def show(self, _is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        id_part = "img2img" if is_img2img else "txt2img"
        artist_chain = _ARTIST_COMPONENTS.get(id_part)
        if artist_chain is None:
            artist_chain = gr.Textbox(
                label="Artist Chain",
                elem_id=f"{id_part}_anima_artist_chain_hidden",
                visible=False,
            )

        startup_enabled = _read_startup_default(is_img2img)

        with InputAccordion(startup_enabled, label=TITLE, elem_id=f"{id_part}_anima_artist_crossattn") as enabled:
            with enabled.extra():
                startup_toggle = gr.Button(
                    value=_startup_button_label(startup_enabled),
                    variant="secondary",
                    elem_id=f"{id_part}_anima_artist_startup_toggle",
                )

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
                strength = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=1.0,
                    step=0.01,
                    label="Strength",
                )

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

            startup_toggle.click(
                fn=_toggle_startup_default_common,
                inputs=[],
                outputs=[startup_toggle],
                show_progress=False,
            )

        self.infotext_fields = [
            (artist_chain, "Anima Artist Chain"),
            (enabled, "AnimaArtistCrossAttn Enabled"),
            (strength, "AnimaArtistCrossAttn Strength"),
            (combine_mode, "AnimaArtistCrossAttn Combine"),
            (fusion_mode, "AnimaArtistCrossAttn Fusion"),
            (apply_to_uncond, "AnimaArtistCrossAttn Apply Uncond"),
            (normalize_weights, "AnimaArtistCrossAttn Normalize Weights"),
            (start_block, "AnimaArtistCrossAttn Start Block"),
            (end_block, "AnimaArtistCrossAttn End Block"),
            (start_percent, "AnimaArtistCrossAttn Start Percent"),
            (end_percent, "AnimaArtistCrossAttn End Percent"),
            (layer_filter, "AnimaArtistCrossAttn Layer Filter"),
        ]
        self.paste_field_names = [name for _, name in self.infotext_fields]

        return [
            artist_chain,
            enabled,
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
        ]

    def process_before_every_sampling(
        self,
        p,
        artist_chain: str,
        enabled: bool,
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
        **kwargs,
    ) -> None:
        if not enabled:
            return

        artists = _split_artist_chain(artist_chain)
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

        if combine_mode not in COMBINE_CHOICES:
            combine_mode = COMBINE_OUTPUT_AVG
        if fusion_mode not in FUSION_CHOICES:
            fusion_mode = FUSION_INTERPOLATE

        start_percent = _clamp01(start_percent)
        end_percent = _clamp01(end_percent)

        base_prompts = _current_base_prompts(p, kwargs)
        individuals, individual_masks = _encode_artist_conditionings(p.sd_model, artists, base_prompts)

        unet = p.sd_model.forge_objects.unet
        patched_unet = _patch_unet(
            unet,
            individuals,
            individual_masks,
            combine_mode,
            fusion_mode,
            float(strength),
            bool(apply_to_uncond),
            bool(normalize_weights),
            int(start_block),
            int(end_block),
            start_percent,
            end_percent,
            str(layer_filter or ""),
        )
        p.sd_model.forge_objects.unet = patched_unet

        _record_generation_params(
            p,
            artists,
            combine_mode,
            fusion_mode,
            float(strength),
            bool(apply_to_uncond),
            bool(normalize_weights),
            int(start_block),
            int(end_block),
            start_percent,
            end_percent,
            str(layer_filter or "").strip(),
        )
