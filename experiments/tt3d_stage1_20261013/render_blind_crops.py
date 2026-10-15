"""Magnify fixed central ROIs of original review frames, without model outputs."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pack", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
manifest = json.loads((args.pack / "manifest.json").read_text(encoding="utf-8"))
args.output.mkdir(parents=True, exist_ok=False)
records = []
for number, case in enumerate(manifest["cases"], 1):
    # Fixed central ROI for every case, irrespective of detector output.
    with Image.open(args.pack / case["image_url"]) as source:
        source = source.convert("RGB")
        w,h = source.size
        box = (round(w*.15), round(h*.22), round(w*.9), round(h*.78))
        crop = source.crop(box)
        # Preserve original pixel scale. Add coordinate rulers outside the ROI.
        canvas = Image.new("RGB", (crop.width+60,crop.height+50), "#202830")
        canvas.paste(crop, (60,50))
        draw = ImageDraw.Draw(canvas)
        draw.text((6,6),f"{number:02d} {case['case_id']} raw pixels; origin=({box[0]}, {box[1]})",fill="white")
        for x in range(((box[0]+49)//50)*50,box[2],50):
            display_x = 60+x-box[0]
            draw.line((display_x,36,display_x,49),fill="white")
            draw.text((display_x-10,23),str(x),fill="white")
        for y in range(((box[1]+49)//50)*50,box[3],50):
            display_y = 50+y-box[1]
            draw.line((48,display_y,59,display_y),fill="white")
            draw.text((4,display_y-5),str(y),fill="white")
        name=f"{number:02d}_{case['case_id']}.png"
        canvas.save(args.output/name)
        records.append({"case_id":case["case_id"],"roi_original_xyxy":box,
                        "crop_pixel_offset":[60,50],"pixel_scale":1,"path":name})
(args.output/"crop_mapping.json").write_text(json.dumps(records,indent=2)+"\n",encoding="utf-8")
print(f"Saved {len(records)} original-scale, unmarked crops")
