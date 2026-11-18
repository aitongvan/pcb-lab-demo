"""Animated exports for the PCB detection interface."""
import io

from PIL import Image, ImageDraw


def scan_gif(original_png, result_png):
    """Reveal an existing prediction with a scan, then hold the complete result."""
    with Image.open(io.BytesIO(original_png)) as image:
        original = image.convert('RGB')
    with Image.open(io.BytesIO(result_png)) as image:
        result = image.convert('RGB')
    if original.size != result.size:
        raise ValueError('Ảnh đầu vào và ảnh kết quả phải cùng kích thước.')

    original.thumbnail((960, 960), Image.Resampling.LANCZOS)
    result = result.resize(original.size, Image.Resampling.LANCZOS)
    width, height = original.size
    frames = []
    glow_height = max(12, height // 14)
    line_width = max(2, height // 320)

    for step in range(25):
        scan_y = round((height - 1) * step / 24)
        frame = original.copy()
        if scan_y > 0:
            frame.paste(result.crop((0, 0, width, scan_y)), (0, 0))
        overlay = Image.new('RGBA', frame.size)
        draw = ImageDraw.Draw(overlay)
        for offset in range(glow_height):
            alpha = round(65 * (1 - offset / glow_height) ** 2)
            y = scan_y - offset
            draw.line((0, y, width - 1, y), fill=(0, 219, 225, alpha))
        draw.line((0, scan_y, width - 1, scan_y), fill=(0, 165, 185, 255), width=line_width + 2)
        draw.line((0, scan_y, width - 1, scan_y), fill=(157, 255, 249, 255), width=line_width)
        frame = Image.alpha_composite(frame.convert('RGBA'), overlay).convert('RGB')
        frames.append(frame.quantize(colors=192))

    frames.append(result.quantize(colors=192))
    output = io.BytesIO()
    frames[0].save(output, format='GIF', save_all=True, append_images=frames[1:],
                   duration=[60] * 25 + [1400], disposal=2, optimize=False)
    return output.getvalue()
