import os
import shutil

from PIL import Image
from omg import WAD
import struct

from tqdm import tqdm


def parse_texture_header(data):
    num_textures = struct.unpack_from("<I", data, 0)[0]
    offsets = [
        struct.unpack_from("<I", data, 4 + i * 4)[0] for i in range(num_textures)
    ]
    textures = []
    for offset in offsets:
        name = data[offset : offset + 8].rstrip(b"\0").decode("ascii")
        width, height = struct.unpack_from("<HH", data, offset + 12)
        patch_count = struct.unpack_from("<H", data, offset + 20)[0]
        patches = []
        for i in range(patch_count):
            patch_offset = offset + 22 + i * 10
            x_off, y_off, patch_id, _, _ = struct.unpack_from(
                "<hhHhh", data, patch_offset
            )
            patches.append({"x": x_off, "y": y_off, "patch_id": patch_id})
        textures.append(
            {"name": name, "width": width, "height": height, "patches": patches}
        )
    return textures


def parse_pnames(data):
    num = struct.unpack_from("<I", data, 0)[0]
    names = []
    for i in range(num):
        offset = 4 + i * 8
        raw = data[offset : offset + 8]
        name = raw.rstrip(b"\0").decode("ascii")
        names.append(name)
    return names


def load_palette(wad):
    lump = wad.palette.bytes
    pal = lump[:768]
    return [tuple(pal[i : i + 3]) for i in range(0, 768, 3)]


def extract_patch(data):
    width, height, left, top = struct.unpack_from("<HHhh", data, 0)
    column_offsets = [
        struct.unpack_from("<I", data, 8 + i * 4)[0] for i in range(width)
    ]
    pixels = [[255] * height for _ in range(width)]

    for x in range(width):
        offset = column_offsets[x]
        while True:
            top_delta = data[offset]
            if top_delta == 255:
                break
            count = data[offset + 1]
            offset += 3
            for i in range(count):
                y = top_delta + i
                pixels[x][y] = data[offset]
                offset += 1
            offset += 1  # skip trailing byte
    return width, height, pixels


def save_texture(texture, patches, pnames, palette):
    img = Image.new("RGB", (texture["width"], texture["height"]), (0, 0, 0))
    for patch in texture["patches"]:
        patch_name = pnames[patch["patch_id"]]
        if patch_name not in patches:
            continue
        w, h, pixels = patches[patch_name]
        for x in range(w):
            for y in range(h):
                if pixels[x][y] == 255:
                    continue
                px = patch["x"] + x
                py = patch["y"] + y
                if 0 <= px < img.width and 0 <= py < img.height:
                    img.putpixel((px, py), palette[pixels[x][y]])

    os.makedirs("data_test", exist_ok=True)
    img.save(f"data_test/{texture['name']}.png")


def extract_doom_texture(wad_path, output_path):
    wad = WAD(wad_path)

    print("loading pnames...")
    pnames = parse_pnames(wad.txdefs["PNAMES"].data)

    textures = parse_texture_header(wad.txdefs["TEXTURE1"].data)

    print("loading palette...")
    palette = load_palette(wad)

    print("loading patches...")
    patches = {name: extract_patch(wad.patches[name].data) for name in wad.patches}

    print("extracting textures...")
    for texture in tqdm(textures, total=len(textures)):
        save_texture(texture, patches, pnames, palette)


# Example usage:
files = os.listdir("./wads")
wads_in_wad_dir = [f for f in files if f.endswith(".wad")]
for wad_file in tqdm(wads_in_wad_dir, total=len(wads_in_wad_dir)):
    print(f"extracting {wad_file}...")
    try:
        extract_doom_texture(os.path.join("./wads", wad_file), "data_test")
    except Exception as e:
        print(f"Error extracting {wad_file}: {e}")
        # move wad to dead_letter_wads directory
        os.makedirs("dead_letter_wads", exist_ok=True)
        shutil.move(
            os.path.join("./wads", wad_file), os.path.join("dead_letter_wads", wad_file)
        )
        continue
