# Anima Artist Mixer for SD Forge Neo

## Credits

This extension is a Forge Neo port inspired by the original ComfyUI project:

**Anima-Artist-Mixer** by **An1X3R**  
https://github.com/An1X3R/Anima-Artist-Mixer

All respect and credit for the original idea and ComfyUI implementation goes to An1X3R. This Forge Neo extension exists as a compatibility port for users who want a similar workflow inside SD Forge Neo.

## What It Does

Anima Artist Mixer adds a dedicated **Artist Chain** prompt box for Anima models. Use it to apply one or more artist/style tags separately from the main positive prompt.

If the **Artist Chain** box is empty, the extension does nothing and generation runs normally.

## Requirements

- SD WebUI Forge Neo
- An Anima model/checkpoint
- This extension installed under:

```text
extensions/sd-forge-anima-artist-mixer/
```

Restart the WebUI after installing or updating the extension.

## Basic Usage

1. Select an Anima model.
2. Write your normal prompt in the positive prompt box.
3. Write your negative prompt as usual.
4. Enter artist/style tags in the **Artist Chain** box.
5. Generate.

Example:

```text
@artist_name, @another_artist
```

Weighted example:

```text
(@artist_name:1.2), (@another_artist:0.8)
```

Artist entries are separated by commas. A trailing comma is ignored.

You can also use v24-style injection weights:

```text
::@artist_name::1.5, @another_artist::0.8
```

The two weight styles affect different parts of the result:

- `(@artist_name:1.2)` changes the text encoder emphasis.
- `::@artist_name::1.5` changes the artist cross-attention injection weight.

The forms can be combined:

```text
::(@artist_name:1.2)::1.5
```

Injection weights are clamped to `0.0` through `4.0`. If any artist entry uses an explicit `::weight`, **Normalize Weights** is bypassed for that generation.

<img width="799" height="311" alt="image" src="https://github.com/user-attachments/assets/ee3cf6e8-84d1-458c-89d8-1e3f7c5ec452" />


## Artist Chain Box Position

By default, the **Artist Chain** box appears between the positive prompt and negative prompt.

You can change this in:

```text
Settings > Anima Artist Mixer > Artist Chain textbox position
```

Available positions:

- `between`
- `above`
- `below`

Changing this setting requires reloading the UI.

## AnimaArtistCrossAttn Panel

The **AnimaArtistCrossAttn** panel contains optional controls.

<img width="541" height="397" alt="image" src="https://github.com/user-attachments/assets/d6147586-e789-49a6-bd91-5a5067e7c4f8" />

The extension is always available. There is no ON/OFF switch. It only activates when the **Artist Chain** box contains at least one valid artist/style tag.

Recommended defaults:

- **Combine Mode**: `output_avg`
- **Fusion Mode**: `interpolate`
- **Strength**: `1.0`
- **Normalize Weights**: enabled
- **Apply to Uncond**: disabled

For most users, only **Combine Mode** and **Strength** need adjustment.

## Combine Modes

### output_avg

Good general-purpose mode for mixing multiple artists when not using Forge Couple.

### concat

Combines artist conditionings in one context. This is usually the better choice when using Forge Couple.

### lowrank_avg

Experimental v24 mode that averages artist influence through a low-rank approximation. Use **Lowrank K** in Advanced to control the rank.

## Fusion Modes

### interpolate

Default mode. Blends from the base prompt output toward the artist output.

### concat_with_base

Keeps the base prompt context together with the artist context. This can be useful when the artist effect is too detached from the main prompt.

### base_preserve

Injects artist influence while trying to preserve the base prompt direction. This is useful when style strength should increase without pulling the image too far away from the main prompt.

## Strength

**Strength** supports `0.0` through `4.0`.

- `0.0`: no artist influence
- `1.0`: normal artist influence
- Above `1.0`: extrapolated stronger artist influence

High values can strongly change composition or image quality.

## Advanced Options

- **Lowrank K**: rank used by `lowrank_avg`. If it is greater than or equal to the artist count, the extension falls back to normal `output_avg` behavior.
- **Artist EMA Alpha**: smooths artist influence across sampling steps. Higher values are more stable but can reduce responsiveness.
- **Experimental: Artist Static Capture**: captures the first few unique sampling steps and then reuses the averaged artist influence. It is used with `output_avg` and `lowrank_avg`.
- **Experimental: Static Capture K**: number of unique sampling steps used by Artist Static Capture.
- **Experimental: Deferred Cache**: local speed/quality tradeoff for `output_avg`.
- **Experimental: Merge Similar Artists**: merges very similar artist conditionings. This can change the result noticeably.

If both **Experimental: Artist Static Capture** and **Experimental: Deferred Cache** are enabled, Artist Static Capture takes priority.

## Forge Couple Notes

When using Forge Couple together with Anima Artist Mixer, use:

```text
Combine Mode: concat
```

`output_avg` may make the artist effect very weak with Forge Couple because Forge Couple's regional prompts remain dominant.

Cache-based stabilizers are automatically disabled when Forge Couple is detected.

## LoRA Notes

Artist tags added by an active LoRA can be used in the **Artist Chain** box, but LoRAs should be activated from the normal positive prompt or Forge's network controls.

Do not rely on the **Artist Chain** box as a LoRA activation field.

Example:

```text
Positive Prompt:
<lora:my_artist_lora:1>, 1girl, dress

Artist Chain:
@trained_artist_tag
```

## Infotext

When Artist Chain is active, generation metadata records the Artist Chain and AnimaArtistCrossAttn settings used for the image.

If Artist Chain is empty, no Anima Artist Mixer metadata is added.
