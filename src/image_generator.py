from PIL import Image, ImageDraw, ImageFont
import os
import textwrap

def create_cofrade_card(title, subtitle, output_path="generated/poster.png"):

    width = 1080
    height = 1350

    color_bg = (46, 15, 37)
    color_gold = (197, 160, 89)
    color_text = (250, 245, 230)

    img = Image.new("RGB", (width, height), color=color_bg)
    draw = ImageDraw.Draw(img)

    margin = 40

    draw.rectangle(
        [margin, margin, width-margin, height-margin],
        outline=color_gold,
        width=5
    )

    draw.rectangle(
        [margin+10, margin+10, width-margin-10, height-margin-10],
        outline=color_gold,
        width=2
    )

    font_path = "fonts/Cinzel-Regular.ttf"

    try:
        font_title = ImageFont.truetype(font_path, 80)
        font_sub = ImageFont.truetype(font_path, 50)

    except:
        print("⚠️ Fuente no encontrada.")
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()

    wrapped_title = "\n".join(
        textwrap.wrap(title.upper(), width=18)
    )

    wrapped_sub = "\n".join(
        textwrap.wrap(subtitle, width=25)
    )

    draw.text(
        (width/2, height/3),
        wrapped_title,
        fill=color_text,
        font=font_title,
        anchor="mm"
    )

    draw.text(
        (width/2, height/2 + 50),
        wrapped_sub,
        fill=color_gold,
        font=font_sub,
        anchor="mm"
    )

    draw.text(
        (width/2, height - 100),
        "✝",
        fill=color_gold,
        font=font_title,
        anchor="mm"
    )

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    img.save(
        output_path,
        optimize=True,
        quality=85
    )

    return output_path
