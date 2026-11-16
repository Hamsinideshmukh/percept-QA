"""Prompt builder: turns frames + cues into the chat message format the model was fine-tuned on.
EGO -> one plain sentence in the SYSTEM prompt; TIME -> a label before each frame; zooms -> extra images after the frames."""
from PIL import Image

PREAMBLE = "5 frames from your car's front camera, 1 second apart, oldest first. The last frame is now."
STYLE = "You are the driver. Answer with one short sentence, no preamble."
INSTR = "Answer in one short sentence."
CUE_SET = ("TIME", "EGO", "LIGHT_ZOOM", "FAR_ZOOM")
IM = lambda im: {"type": "image", "image": im}
TX = lambda t: {"type": "text", "text": t}

def frame(clip_dir, i):
    im = Image.open(f"{clip_dir}/{i}.jpg").convert("RGB"); im.thumbnail((512, 512) if i < 4 else (768, 768)); return im

def crop(clip_dir, cx, cy, wc, hc, out):
    img = Image.open(f"{clip_dir}/4.jpg").convert("RGB"); W, H = img.size; wc, hc = min(wc, W), min(hc, H)
    x0 = int(min(max(cx - wc / 2, 0), W - wc)); y0 = int(min(max(cy - hc / 2, 0), H - hc))
    return img.crop((x0, y0, x0 + wc, y0 + hc)).resize(out, Image.LANCZOS)

def build(clip_dir, question, route_info, cues, use=CUE_SET, why_gets_zooms=True):
    """cues = {"state": str|None, "lights": [boxes], "vp": [x, y]}. Returns (messages, list of cues attached, zoom images)."""
    routed = set(route_info["cues"]); given, zooms, c, sysm = [], [], [], None
    if why_gets_zooms and route_info["cls"] in (1, 2): routed |= {"LIGHT_ZOOM", "FAR_ZOOM"}
    if "TIME" in use:
        c.append(TX(PREAMBLE)); given.append("TIME")
        for i in range(5): c += [TX(f"t = {i - 4} s" + (" (now)" if i == 4 else "") + ":"), IM(frame(clip_dir, i))]
    else:
        c += [IM(frame(clip_dir, i)) for i in range(5)]
    if "LIGHT_ZOOM" in use and "LIGHT_ZOOM" in routed and cues.get("lights"):
        for b in cues["lights"][:2]:
            x0, y0, x1, y1 = b["box"]; side = max(240, 7 * max(x1 - x0, y1 - y0))
            z = crop(clip_dir, (x0 + x1) / 2, (y0 + y1) / 2, side, side, (336, 336)); zooms.append(("traffic light", z))
            c += [TX("Zoomed view of a possible traffic light in the last frame:"), IM(z)]
        given.append("LIGHT_ZOOM")
    if "FAR_ZOOM" in use and "FAR_ZOOM" in routed and "LIGHT_ZOOM" not in given and cues.get("vp"):
        z = crop(clip_dir, cues["vp"][0], cues["vp"][1], 960, 540, (448, 252)); zooms.append(("road ahead", z))
        c += [TX("Zoomed view of the road far ahead in the last frame:"), IM(z)]; given.append("FAR_ZOOM")
    if "EGO" in use and "EGO_DETAIL" in routed and cues.get("state"):
        sysm = f"{STYLE}\nRight now your car is {cues['state']}."; given.append("EGO")
    msgs = ([{"role": "system", "content": [TX(sysm)]}] if sysm else []) + [{"role": "user", "content": c + [TX(f"{question}\n{INSTR}")]}]
    return msgs, given, zooms
