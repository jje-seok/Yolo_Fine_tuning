from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageOps, ImageTk

from .data import IMAGE_EXTS


@dataclass
class ImageItem:
    image_path: Path
    label_path: Path
    split: str


@dataclass
class LabelBox:
    cx: float
    cy: float
    w: float
    h: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual visual review and correction tool for YOLO person labels.")
    parser.add_argument("--dataset", default="calibration/labeled/front", help="YOLO dataset root.")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"], choices=["train", "valid", "test"])
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--max-height", type=int, default=820)
    parser.add_argument("--start", default=None, help="Start index or image filename.")
    parser.add_argument("--resume", action="store_true", help="Start from the first image not marked accepted/edited.")
    parser.add_argument("--backup-dir", default=None, help="Optional label backup directory.")
    return parser.parse_args()


def collect_items(dataset_root: Path, splits: list[str]) -> list[ImageItem]:
    items: list[ImageItem] = []
    for split in splits:
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        if not image_dir.exists():
            continue
        for image_path in sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
            rel = image_path.relative_to(image_dir)
            label_path = label_dir / rel.with_suffix(".txt")
            items.append(ImageItem(image_path=image_path, label_path=label_path, split=split))
    if not items:
        raise FileNotFoundError(f"no images found under {dataset_root / 'images'}")
    return items


def load_person_labels(label_path: Path) -> list[LabelBox]:
    if not label_path.exists():
        return []
    boxes: list[LabelBox] = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            cx, cy, w, h = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if cls_id != 0:
            continue
        if w <= 0 or h <= 0:
            continue
        boxes.append(
            LabelBox(
                cx=min(max(cx, 0.0), 1.0),
                cy=min(max(cy, 0.0), 1.0),
                w=min(max(w, 0.0), 1.0),
                h=min(max(h, 0.0), 1.0),
            )
        )
    return boxes


def save_person_labels(label_path: Path, boxes: list[LabelBox]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"0 {b.cx:.6f} {b.cy:.6f} {b.w:.6f} {b.h:.6f}" for b in boxes]
    label_path.write_text("\n".join(lines))


class LabelReviewApp:
    def __init__(
        self,
        root: tk.Tk,
        dataset_root: Path,
        items: list[ImageItem],
        max_width: int,
        max_height: int,
        backup_dir: Path,
        start_index: int = 0,
    ) -> None:
        self.root = root
        self.dataset_root = dataset_root
        self.items = items
        self.max_width = max_width
        self.max_height = max_height
        self.backup_dir = backup_dir
        self.review_manifest = dataset_root / "label_review_manifest.csv"
        self.reviewed = self.load_review_manifest()
        self.backed_up: set[Path] = set()

        self.index = start_index
        self.boxes: list[LabelBox] = []
        self.selected: int | None = None
        self.dirty = False
        self.current_image: Image.Image | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.display_scale = 1.0
        self.drag_start: tuple[int, int] | None = None
        self.drag_rect: int | None = None

        self.root.title("YOLO person label review")
        self.status = tk.Label(root, anchor="w", font=("Arial", 11))
        self.status.pack(fill="x", padx=8, pady=(8, 4))

        self.canvas = tk.Canvas(root, bg="#1f1f1f", highlightthickness=0)
        self.canvas.pack(padx=8, pady=4)

        controls = tk.Frame(root)
        controls.pack(fill="x", padx=8, pady=(4, 8))
        tk.Button(controls, text="Prev", command=self.previous_image).pack(side="left")
        tk.Button(controls, text="OK / Next", command=self.accept_and_next).pack(side="left", padx=4)
        tk.Button(controls, text="Save", command=self.save_current).pack(side="left", padx=4)
        tk.Button(controls, text="Delete Box", command=self.delete_selected).pack(side="left", padx=4)
        tk.Button(controls, text="Clear", command=self.clear_boxes).pack(side="left", padx=4)
        tk.Button(controls, text="Quit", command=self.quit).pack(side="right")

        self.help_text = tk.Label(
            root,
            anchor="w",
            text=(
                "Drag mouse: add person box | Click box: select | Enter/n/Right: OK next | "
                "Delete: remove selected | u: undo | c: clear | s: save | p/Left: previous | q/Esc: quit"
            ),
        )
        self.help_text.pack(fill="x", padx=8, pady=(0, 8))

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.root.bind("<Return>", lambda _event: self.accept_and_next())
        self.root.bind("n", lambda _event: self.accept_and_next())
        self.root.bind("<Right>", lambda _event: self.accept_and_next())
        self.root.bind("p", lambda _event: self.previous_image())
        self.root.bind("<Left>", lambda _event: self.previous_image())
        self.root.bind("s", lambda _event: self.save_current())
        self.root.bind("u", lambda _event: self.undo_box())
        self.root.bind("c", lambda _event: self.clear_boxes())
        self.root.bind("<Delete>", lambda _event: self.delete_selected())
        self.root.bind("<BackSpace>", lambda _event: self.delete_selected())
        self.root.bind("q", lambda _event: self.quit())
        self.root.bind("<Escape>", lambda _event: self.quit())

        self.load_image()

    def load_review_manifest(self) -> dict[str, dict[str, str]]:
        if not self.review_manifest.exists():
            return {}
        rows = csv.DictReader(self.review_manifest.open(newline=""))
        return {row["key"]: row for row in rows}

    def item_key(self, item: ImageItem) -> str:
        return str(item.image_path.relative_to(self.dataset_root))

    def write_review_manifest(self) -> None:
        fieldnames = ["key", "split", "image", "label", "status", "num_labels", "updated_at"]
        with self.review_manifest.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for item in self.items:
                key = self.item_key(item)
                if key in self.reviewed:
                    writer.writerow(self.reviewed[key])

    def mark_reviewed(self, status: str) -> None:
        item = self.items[self.index]
        key = self.item_key(item)
        self.reviewed[key] = {
            "key": key,
            "split": item.split,
            "image": item.image_path.name,
            "label": str(item.label_path.relative_to(self.dataset_root)),
            "status": status,
            "num_labels": str(len(self.boxes)),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.write_review_manifest()

    def current_review_status(self) -> str | None:
        return self.reviewed.get(self.item_key(self.items[self.index]), {}).get("status")

    def load_image(self) -> None:
        item = self.items[self.index]
        self.current_image = ImageOps.exif_transpose(Image.open(item.image_path)).convert("RGB")
        width, height = self.current_image.size
        self.display_scale = min(self.max_width / width, self.max_height / height, 1.0)
        display_size = (int(width * self.display_scale), int(height * self.display_scale))
        resized = self.current_image.resize(display_size, Image.BILINEAR)
        self.photo = ImageTk.PhotoImage(resized)

        self.canvas.config(width=display_size[0], height=display_size[1])
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

        self.boxes = load_person_labels(item.label_path)
        self.selected = None
        self.dirty = False
        self.drag_start = None
        self.drag_rect = None
        self.redraw_boxes()
        self.update_status()

    def update_status(self) -> None:
        item = self.items[self.index]
        key = self.item_key(item)
        reviewed = self.reviewed.get(key, {}).get("status", "unreviewed")
        dirty = " modified" if self.dirty else ""
        self.status.config(
            text=(
                f"{self.index + 1}/{len(self.items)} | split={item.split} | "
                f"{item.image_path.name} | boxes={len(self.boxes)} | {reviewed}{dirty}"
            )
        )

    def box_to_canvas(self, box: LabelBox) -> tuple[float, float, float, float]:
        assert self.current_image is not None
        width, height = self.current_image.size
        x1 = (box.cx - box.w / 2) * width * self.display_scale
        y1 = (box.cy - box.h / 2) * height * self.display_scale
        x2 = (box.cx + box.w / 2) * width * self.display_scale
        y2 = (box.cy + box.h / 2) * height * self.display_scale
        return x1, y1, x2, y2

    def redraw_boxes(self) -> None:
        self.canvas.delete("box")
        for idx, box in enumerate(self.boxes):
            x1, y1, x2, y2 = self.box_to_canvas(box)
            color = "#ff3b30" if idx == self.selected else "#00e676"
            width = 3 if idx == self.selected else 2
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=width, tags="box")
            self.canvas.create_text(x1 + 4, y1 + 4, anchor="nw", text="person", fill=color, tags="box")
        self.update_status()

    def find_box_at(self, x: int, y: int) -> int | None:
        for idx in range(len(self.boxes) - 1, -1, -1):
            x1, y1, x2, y2 = self.box_to_canvas(self.boxes[idx])
            if x1 <= x <= x2 and y1 <= y <= y2:
                return idx
        return None

    def clamp_canvas_point(self, x: int, y: int) -> tuple[int, int]:
        canvas_w = int(self.canvas.cget("width"))
        canvas_h = int(self.canvas.cget("height"))
        return min(max(x, 0), canvas_w), min(max(y, 0), canvas_h)

    def on_mouse_down(self, event: tk.Event) -> None:
        x, y = self.clamp_canvas_point(int(event.x), int(event.y))
        selected = self.find_box_at(x, y)
        if selected is not None:
            self.selected = selected
            self.drag_start = None
            self.redraw_boxes()
            return
        self.selected = None
        self.drag_start = (x, y)
        if self.drag_rect is not None:
            self.canvas.delete(self.drag_rect)
        self.drag_rect = self.canvas.create_rectangle(x, y, x, y, outline="#ffeb3b", width=2)
        self.redraw_boxes()

    def on_mouse_drag(self, event: tk.Event) -> None:
        if self.drag_start is None or self.drag_rect is None:
            return
        x, y = self.clamp_canvas_point(int(event.x), int(event.y))
        x0, y0 = self.drag_start
        self.canvas.coords(self.drag_rect, x0, y0, x, y)

    def on_mouse_up(self, event: tk.Event) -> None:
        if self.drag_start is None:
            return
        assert self.current_image is not None
        x, y = self.clamp_canvas_point(int(event.x), int(event.y))
        x0, y0 = self.drag_start
        self.drag_start = None
        if self.drag_rect is not None:
            self.canvas.delete(self.drag_rect)
            self.drag_rect = None

        x1, x2 = sorted((x0, x))
        y1, y2 = sorted((y0, y))
        if x2 - x1 < 5 or y2 - y1 < 5:
            self.redraw_boxes()
            return

        image_w, image_h = self.current_image.size
        ix1 = x1 / self.display_scale
        iy1 = y1 / self.display_scale
        ix2 = x2 / self.display_scale
        iy2 = y2 / self.display_scale
        box = LabelBox(
            cx=((ix1 + ix2) / 2) / image_w,
            cy=((iy1 + iy2) / 2) / image_h,
            w=(ix2 - ix1) / image_w,
            h=(iy2 - iy1) / image_h,
        )
        self.boxes.append(box)
        self.selected = len(self.boxes) - 1
        self.dirty = True
        self.redraw_boxes()

    def backup_label_if_needed(self, item: ImageItem) -> None:
        if item.label_path in self.backed_up:
            return
        backup_path = self.backup_dir / item.split / item.label_path.name
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        if item.label_path.exists():
            shutil.copy2(item.label_path, backup_path)
        else:
            backup_path.with_suffix(".missing").write_text("")
        self.backed_up.add(item.label_path)

    def save_current(self) -> None:
        item = self.items[self.index]
        had_changes = self.dirty
        if had_changes:
            self.backup_label_if_needed(item)
        save_person_labels(item.label_path, self.boxes)
        self.dirty = False
        status = "edited" if had_changes or self.current_review_status() == "edited" else "accepted"
        self.mark_reviewed(status)
        self.update_status()

    def accept_and_next(self) -> None:
        if self.dirty:
            self.save_current()
        else:
            if self.current_review_status() != "edited":
                self.mark_reviewed("accepted")
        if self.index + 1 >= len(self.items):
            messagebox.showinfo("Done", "All images reviewed.")
            self.update_status()
            return
        self.index += 1
        self.load_image()

    def previous_image(self) -> None:
        if self.dirty:
            self.save_current()
        if self.index > 0:
            self.index -= 1
            self.load_image()

    def delete_selected(self) -> None:
        if not self.boxes:
            return
        index = self.selected if self.selected is not None else len(self.boxes) - 1
        del self.boxes[index]
        self.selected = None
        self.dirty = True
        self.redraw_boxes()

    def undo_box(self) -> None:
        if not self.boxes:
            return
        self.boxes.pop()
        self.selected = None
        self.dirty = True
        self.redraw_boxes()

    def clear_boxes(self) -> None:
        if not self.boxes:
            return
        self.boxes = []
        self.selected = None
        self.dirty = True
        self.redraw_boxes()

    def quit(self) -> None:
        if self.dirty and messagebox.askyesno("Save changes?", "Save current label changes before quitting?"):
            self.save_current()
        self.root.destroy()


def find_start_index(items: list[ImageItem], start: str | None, reviewed: dict[str, dict[str, str]], dataset_root: Path) -> int:
    if start:
        if start.isdigit():
            return min(max(int(start), 0), len(items) - 1)
        for index, item in enumerate(items):
            if item.image_path.name == start or str(item.image_path.relative_to(dataset_root)) == start:
                return index
        raise ValueError(f"start image not found: {start}")

    accepted = {"accepted", "edited"}
    for index, item in enumerate(items):
        key = str(item.image_path.relative_to(dataset_root))
        if reviewed.get(key, {}).get("status") not in accepted:
            return index
    return 0


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset)
    items = collect_items(dataset_root, args.splits)
    backup_dir = Path(args.backup_dir) if args.backup_dir else dataset_root / "label_review_backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
    existing_reviewed = {}
    manifest = dataset_root / "label_review_manifest.csv"
    if manifest.exists():
        existing_reviewed = {row["key"]: row for row in csv.DictReader(manifest.open(newline=""))}
    start_index = find_start_index(items, args.start, existing_reviewed if args.resume else {}, dataset_root)

    root = tk.Tk()
    LabelReviewApp(root, dataset_root, items, args.max_width, args.max_height, backup_dir, start_index=start_index)
    root.mainloop()


if __name__ == "__main__":
    main()
