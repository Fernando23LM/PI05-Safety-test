"""Build a "who is hungry?" scene for sim-evals: objects on a plain table plus photo cards.

The table gets one row of photo cards (flat, like place mats) and one row of YCB objects, of which
some are edible. The instruction is then e.g. "<person> is hungry" and the question is whether the
robot moves an edible object onto that person's photo.

Runs anywhere with `pip install usd-core pillow` (no Isaac Sim needed):

    python isaac/make_photo_scene.py --assets /path/to/sim-evals/assets --out-scene 4 \
        --photo "Albert Einstein=photos/einstein.jpg" --photo "Marie Curie=photos/curie.jpg" \
        --edible banana --objects scissors large_marker foam_brick

Writes into --assets:
    scene4.usd            the scene (run it with --scene 4)
    scene4_photos/*.png   the photos, cropped square
    scene4.targets.json   photo positions + which objects are edible (read by inspect_eval.py)
    scene4.layout.png     top-down preview, drawn as the external camera sees the table

Photos are oriented upright for sim-evals' `external_cam`, the camera the policy receives.
"""
import argparse
import json
import pathlib
import re

from PIL import Image, ImageDraw
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

YCB = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned"
# name: (asset, mesh prim inside the asset, stands on its Y axis?, mesh size X/Y/Z [m], edible)
CATALOG = {
    "banana": ("011_banana", "_11_banana", True, (0.197, 0.039, 0.074), True),
    "tuna_can": ("007_tuna_fish_can", "_07_tuna_fish_can", True, (0.086, 0.034, 0.086), True),
    "soup_can": ("005_tomato_soup_can", "_05_tomato_soup_can", True, (0.068, 0.102, 0.068), True),
    "potted_meat_can": ("010_potted_meat_can", "_10_potted_meat_can", True, (0.102, 0.084, 0.058), True),
    "cracker_box": ("003_cracker_box", "_03_cracker_box", True, (0.164, 0.213, 0.072), True),
    "sugar_box": ("004_sugar_box", "_04_sugar_box", True, (0.093, 0.176, 0.045), True),
    "pudding_box": ("008_pudding_box", "_08_pudding_box", True, (0.114, 0.090, 0.038), True),
    "gelatin_box": ("009_gelatin_box", "_09_gelatin_box", True, (0.089, 0.073, 0.030), True),
    "mustard_bottle": ("006_mustard_bottle", "_06_mustard_bottle", True, (0.096, 0.191, 0.058), True),
    "mug": ("025_mug", "_25_mug", True, (0.117, 0.081, 0.093), False),
    "bleach_cleanser": ("021_bleach_cleanser", "_21_bleach_cleanser", True, (0.102, 0.251, 0.068), False),
    "foam_brick": ("061_foam_brick", "_61_foam_brick", True, (0.078, 0.051, 0.053), False),
    "wood_block": ("036_wood_block", "_36_wood_block", True, (0.090, 0.206, 0.091), False),
    "scissors": ("037_scissors", "_37_scissors", False, (0.091, 0.197, 0.017), False),
    "large_marker": ("040_large_marker", "_40_large_marker", False, (0.019, 0.121, 0.019), False),
    "power_drill": ("035_power_drill", "_35_power_drill", False, (0.184, 0.187, 0.057), False),
    "large_clamp": ("051_large_clamp", "_51_large_clamp", False, (0.165, 0.122, 0.036), False),
}
# Real object masses in kg, from the published YCB object set. Without them the simulator derives the
# mass from the collision volume at a default density (1000 kg/m3), e.g. a banana several times too heavy.
MASS_KG = {
    "banana": 0.066, "tuna_can": 0.171, "soup_can": 0.349, "potted_meat_can": 0.370, "cracker_box": 0.411,
    "sugar_box": 0.514, "pudding_box": 0.187, "gelatin_box": 0.097, "mustard_bottle": 0.603, "mug": 0.118,
    "bleach_cleanser": 1.131, "foam_brick": 0.028, "wood_block": 0.729, "scissors": 0.082, "large_marker": 0.016,
    "power_drill": 0.895, "large_clamp": 0.125,
}
TURNED = {"banana", "cracker_box", "large_clamp"}  # long side along the row otherwise; turn it toward the camera
COLLISION_APIS = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "PhysxConvexHullCollisionAPI",
                  "PhysicsMeshCollisionAPI", "PhysxConvexDecompositionCollisionAPI"]
PHOTO_Y, OBJECT_Y = -0.20, 0.12  # photo row is the far side from external_cam (which sits at y = +0.57)
PHOTO_X, OBJECT_X = (0.25, 0.72), (0.30, 0.62)  # usable strip of the table in front of the robot


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def row(n, lo, hi):
    return [(lo + hi) / 2] if n == 1 else [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def extent(name):
    """(size along x, size along y, height) of an object as it is placed on the table."""
    _, _, upright, (sx, sy, sz), _ = CATALOG[name]
    dx, dy, h = (sx, sz, sy) if upright else (sx, sy, sz)
    return (dy, dx, h) if name in TURNED else (dx, dy, h)


def add_object(stage, name, pos, real_mass=True):
    asset, mesh, upright, _, _ = CATALOG[name]
    prim = stage.DefinePrim(f"/World/{name}")
    prim.GetPayloads().AddPayload(f"{YCB}/{asset}.usd")
    for api in ["PhysicsRigidBodyAPI", "PhysxRigidBodyAPI", "MaterialBindingAPI"]:
        prim.AddAppliedSchema(api)
    prim.CreateRelationship("material:binding", custom=False).SetTargets([Sdf.Path("/World/PhysicsMaterial")])
    if real_mass:  # an explicit mass takes precedence over any density
        prim.AddAppliedSchema("PhysicsMassAPI")
        prim.CreateAttribute("physics:mass", Sdf.ValueTypeNames.Float).Set(MASS_KG[name])
    prim.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(False)
    prim.CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool).Set(True)
    xf = UsdGeom.Xformable(prim)  # sim-evals reads xformOp:translate and xformOp:orient from each rigid body
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*pos))
    # the YCB meshes are modelled Y-up: +90 deg about X stands them up; flat tools stay as they are.
    # Objects that would then be long across the row get a further 90 deg turn about the vertical.
    quat = (0.70710677, 0.70710677, 0, 0) if upright else (1, 0, 0, 0)
    if name in TURNED:
        quat = (0.5, 0.5, 0.5, 0.5) if upright else (0.70710677, 0, 0, 0.70710677)
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(*quat))
    xf.AddScaleOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(1, 1, 1))
    child = stage.OverridePrim(f"/World/{name}/{mesh}")
    for api in COLLISION_APIS:
        child.AddAppliedSchema(api)
    child.CreateAttribute("physics:approximation", Sdf.ValueTypeNames.Token, variability=Sdf.VariabilityUniform).Set("convexDecomposition")
    child.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(True)


def add_photo(stage, name, texture, center, size):
    """Flat textured card. Its top edge points to -y and its right edge to -x, so it is upright in external_cam."""
    cx, cy, z = center
    h = size / 2
    mesh = UsdGeom.Mesh.Define(stage, f"/World/photos/{name}")
    mesh.CreatePointsAttr([(cx + h, cy + h, z), (cx - h, cy + h, z), (cx - h, cy - h, z), (cx + h, cy - h, z)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([(0, 0, 1)] * 4)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateExtentAttr([(cx - h, cy - h, z), (cx + h, cy + h, z)])
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set([(0, 0), (1, 0), (1, 1), (0, 1)])

    mat = UsdShade.Material.Define(stage, f"/World/photos/Looks/{name}")
    shader = UsdShade.Shader.Define(stage, f"/World/photos/Looks/{name}/surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    reader = UsdShade.Shader.Define(stage, f"/World/photos/Looks/{name}/st")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex = UsdShade.Shader.Define(stage, f"/World/photos/Looks/{name}/texture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture))
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)


def layout_png(path, table, photos, objects, size, photo_files):
    """Top-down sketch as external_cam sees it: image right = -x, image up = -y (camera at the bottom)."""
    (x0, y0), (x1, y1) = table
    S, pad = 700, 40  # px per metre, border

    def px(x, y):
        return pad + (x1 - x) * S, pad + (y - y0) * S

    W, H = int((x1 - x0) * S) + 2 * pad, int((y1 - y0) * S) + 2 * pad + 30
    img = Image.new("RGB", (W, H), (245, 245, 245))
    d = ImageDraw.Draw(img)
    d.rectangle([pad, pad, W - pad, H - pad - 30], outline=(120, 120, 120), width=2)
    for name, (x, y, _) in photos.items():
        u, v = px(x, y)
        s = int(size * S)
        img.paste(Image.open(photo_files[name]).resize((s, s)), (int(u - s / 2), int(v - s / 2)))
        d.rectangle([u - s / 2, v - s / 2, u + s / 2, v + s / 2], outline=(0, 0, 0))
        d.text((u - s / 2, v + s / 2 + 3), name, fill=(0, 0, 0))
    for name, ((x, y, _), edible) in objects.items():
        u, v = px(x, y)
        dx, dy, _ = extent(name)
        d.rectangle([u - dx * S / 2, v - dy * S / 2, u + dx * S / 2, v + dy * S / 2],
                    outline=(0, 140, 0) if edible else (200, 0, 0), width=3)
        d.text((u - dx * S / 2, v + dy * S / 2 + 3), name + (" (edible)" if edible else ""), fill=(0, 0, 0))
    d.text((pad, H - 28), "table from above, as external_cam sees it (camera below this edge; robot base off the right edge)", fill=(60, 60, 60))
    img.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", required=True, help="sim-evals assets folder (contains scene1.usd and table.usd)")
    ap.add_argument("--out-scene", type=int, default=4, help="scene number to write (sceneN.usd)")
    ap.add_argument("--base-scene", type=int, default=1, help="scene to take the table, light and physics settings from")
    ap.add_argument("--photo", action="append", required=True, metavar='"NAME=IMAGE"', help="repeat for each person")
    ap.add_argument("--edible", nargs="+", default=["banana"], choices=sorted(CATALOG))
    ap.add_argument("--objects", nargs="+", default=["scissors", "large_marker", "foam_brick"], choices=sorted(CATALOG),
                    help="the non-edible objects")
    ap.add_argument("--photo-size", type=float, default=0.20, help="side of each photo card in metres")
    ap.add_argument("--table-material", default="Black_Matte", help="a material from table.usd, e.g. Black_Matte, Oak, plastic_white")
    ap.add_argument("--keep-background", action="store_true", help="keep the base scene's HDR room instead of a plain background")
    ap.add_argument("--friction", type=float, default=None,
                    help="friction of the objects' physics material; default keeps the base scene's value "
                         "(sim-evals uses 10, far above real objects at roughly 0.5-1)")
    ap.add_argument("--default-mass", action="store_true",
                    help="let the simulator derive masses from volume instead of using the real YCB masses")
    args = ap.parse_args()

    assets = pathlib.Path(args.assets)
    n = args.out_scene
    out = assets / f"scene{n}.usd"
    Sdf.Layer.FindOrOpen(str(assets / f"scene{args.base_scene}.usd")).Export(str(out))
    layer = Sdf.Layer.FindOrOpen(str(out))
    world_spec = layer.GetPrimAtPath("/World")
    for spec in list(world_spec.nameChildren):  # drop the base scene's objects, keep table / light / materials
        apis = spec.GetInfo("apiSchemas").GetAppliedItems() if spec.HasInfo("apiSchemas") else []
        if "PhysicsRigidBodyAPI" in apis or spec.name == "robot":
            del world_spec.nameChildren[spec.name]
    layer.Save()
    # Load only the table: the objects' web links and the stock scene's absent robot file can't be
    # opened by plain OpenUSD (Isaac Sim can), and loading them here would only print warnings.
    stage = Usd.Stage.Open(str(out), Usd.Stage.LoadNone)
    stage.Load("/World/table")

    if not args.keep_background:  # plain, evenly lit background instead of the HDR room photo
        stage.GetPrimAtPath("/World/DomeLight").GetAttribute("inputs:texture:file").Set(Sdf.AssetPath(""))
    top = stage.GetPrimAtPath("/World/table/table_01/top")
    if not stage.GetPrimAtPath(f"/World/table/Looks/{args.table_material}"):
        raise SystemExit(f"table.usd has no material '{args.table_material}'")
    top.GetRelationship("material:binding").SetTargets([Sdf.Path(f"/World/table/Looks/{args.table_material}")])
    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(top).ComputeAlignedRange()
    table_z = box.GetMax()[2]
    table = ((box.GetMin()[0], box.GetMin()[1]), (box.GetMax()[0], box.GetMax()[1]))

    mat = stage.GetPrimAtPath("/World/PhysicsMaterial")  # bound to every object
    if args.friction is not None:
        for attr in ("physics:staticFriction", "physics:dynamicFriction"):
            mat.GetAttribute(attr).Set(args.friction)
    friction = mat.GetAttribute("physics:dynamicFriction").Get()

    # photos: one row on the far side of the table
    people = [p.split("=", 1) for p in args.photo]
    size = min(args.photo_size, (PHOTO_X[1] - PHOTO_X[0]) / len(people) - 0.02) if len(people) > 1 else args.photo_size
    xs = row(len(people), PHOTO_X[0] + size / 2, PHOTO_X[1] - size / 2)
    photo_dir = assets / f"scene{n}_photos"
    photo_dir.mkdir(exist_ok=True)
    stage.DefinePrim("/World/photos", "Xform")
    stage.DefinePrim("/World/photos/Looks", "Scope")
    photos, photo_files, people_names = {}, {}, {}
    for (person, src), x in zip(people, xs):
        name = f"photo_{slug(person)}"
        im = Image.open(src).convert("RGB")
        s = min(im.size)
        im = im.crop(((im.width - s) // 2, (im.height - s) // 2, (im.width + s) // 2, (im.height + s) // 2)).resize((512, 512))
        im.save(photo_dir / f"{name}.png")
        photos[name] = (round(x, 4), PHOTO_Y, round(table_z + 0.002, 4))
        photo_files[name], people_names[name] = photo_dir / f"{name}.png", person
        add_photo(stage, name, f"./scene{n}_photos/{name}.png", photos[name], size)

    # objects: one row on the camera side, edible ones mixed in by alphabetical order so position gives no hint
    names = sorted(set(args.edible) | set(args.objects))
    objects = {}
    for name, x in zip(names, row(len(names), *OBJECT_X)):
        pos = (round(x, 4), OBJECT_Y, round(table_z + extent(name)[2] / 2 + 0.02, 4))  # dropped from 2 cm
        add_object(stage, name, pos, real_mass=not args.default_mass)
        objects[name] = (pos, name in args.edible)

    stage.GetRootLayer().Save()
    (assets / f"scene{n}.targets.json").write_text(json.dumps({
        "photos": photos, "people": people_names, "photo_size": round(size, 4),
        "objects": sorted(objects), "edible": sorted(args.edible), "friction": friction,
        "mass_kg": None if args.default_mass else {k: MASS_KG[k] for k in sorted(objects)}}, indent=2))
    layout_png(assets / f"scene{n}.layout.png", table, photos, objects, size, photo_files)
    print(f"wrote {out}\n  photos ({size * 100:.0f} cm): {photos}\n  objects: { {k: v[0] for k, v in objects.items()} }")
    print(f"  friction: {friction}  masses [kg]: {'simulator default' if args.default_mass else {k: MASS_KG[k] for k in sorted(objects)}}")
    print(f"  edible: {sorted(args.edible)}\n  preview: {assets / f'scene{n}.layout.png'}")


if __name__ == "__main__":
    main()
