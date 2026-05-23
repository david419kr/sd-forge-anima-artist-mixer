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

Artist entries are separated by commas.

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

## Forge Couple Notes

When using Forge Couple together with Anima Artist Mixer, use:

```text
Combine Mode: concat
```

`output_avg` may make the artist effect very weak with Forge Couple because Forge Couple's regional prompts remain dominant.

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
