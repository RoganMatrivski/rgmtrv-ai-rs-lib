use image::{imageops, DynamicImage, GenericImageView, Rgba, RgbaImage};

// ── Operations ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Op {
    // Crop: remove N pixels from the given edge(s)
    CropTop(u32),
    CropBottom(u32),
    CropLeft(u32),
    CropRight(u32),
    CropX(u32),            // left + right each
    CropY(u32),            // top  + bottom each
    CropLongestSide(u32),  // both ends of the longer dimension
    CropShortestSide(u32), // both ends of the shorter dimension

    // Pad: add N pixels of pad_color to the given edge(s)
    PadTop(u32),
    PadBottom(u32),
    PadLeft(u32),
    PadRight(u32),
    PadX(u32),            // left + right each
    PadY(u32),            // top  + bottom each
    PadLongestSide(u32),  // both ends of the longer dimension
    PadShortestSide(u32), // both ends of the shorter dimension

    /// Multiply both dimensions by `factor`
    Scale(f32),

    /// Resize to `(width, height)`; pass `None` to preserve the aspect-ratio
    /// dimension (at least one must be `Some`).
    Resize(Option<u32>, Option<u32>),

    /// Resize so the longest side equals `pixels`; other side scales to match.
    ResizeLongestSide(u32),

    /// Resize so the shortest side equals `pixels`; other side scales to match.
    ResizeShortestSide(u32),
}

// ── Processor ────────────────────────────────────────────────────────────────

#[derive(bon::Builder)]
pub struct ImageProcessor {
    image: DynamicImage,

    /// Background colour used for pad operations. Defaults to opaque black.
    #[builder(default = Rgba([0, 0, 0, 255]))]
    pad_color: Rgba<u8>,

    /// Accumulated operation queue (populated by the builder-style chain).
    #[builder(default)]
    ops: Vec<Op>,
}

// ── Chainable operation methods ───────────────────────────────────────────────

impl ImageProcessor {
    // ── crop ──────────────────────────────────────────────────────────────────

    pub fn crop_top(mut self, pixels: u32) -> Self {
        self.ops.push(Op::CropTop(pixels));
        self
    }
    pub fn crop_bottom(mut self, pixels: u32) -> Self {
        self.ops.push(Op::CropBottom(pixels));
        self
    }
    pub fn crop_left(mut self, pixels: u32) -> Self {
        self.ops.push(Op::CropLeft(pixels));
        self
    }
    pub fn crop_right(mut self, pixels: u32) -> Self {
        self.ops.push(Op::CropRight(pixels));
        self
    }
    /// Crop `pixels` from both the left *and* right edges.
    pub fn crop_x(mut self, pixels: u32) -> Self {
        self.ops.push(Op::CropX(pixels));
        self
    }
    /// Crop `pixels` from both the top *and* bottom edges.
    pub fn crop_y(mut self, pixels: u32) -> Self {
        self.ops.push(Op::CropY(pixels));
        self
    }
    /// Crop `pixels` from both ends of whichever dimension is currently larger.
    pub fn crop_longest_side(mut self, pixels: u32) -> Self {
        self.ops.push(Op::CropLongestSide(pixels));
        self
    }
    /// Crop `pixels` from both ends of whichever dimension is currently smaller.
    pub fn crop_shortest_side(mut self, pixels: u32) -> Self {
        self.ops.push(Op::CropShortestSide(pixels));
        self
    }

    // ── pad ───────────────────────────────────────────────────────────────────

    pub fn pad_top(mut self, pixels: u32) -> Self {
        self.ops.push(Op::PadTop(pixels));
        self
    }
    pub fn pad_bottom(mut self, pixels: u32) -> Self {
        self.ops.push(Op::PadBottom(pixels));
        self
    }
    pub fn pad_left(mut self, pixels: u32) -> Self {
        self.ops.push(Op::PadLeft(pixels));
        self
    }
    pub fn pad_right(mut self, pixels: u32) -> Self {
        self.ops.push(Op::PadRight(pixels));
        self
    }
    /// Add `pixels` padding to both the left *and* right edges.
    pub fn pad_x(mut self, pixels: u32) -> Self {
        self.ops.push(Op::PadX(pixels));
        self
    }
    /// Add `pixels` padding to both the top *and* bottom edges.
    pub fn pad_y(mut self, pixels: u32) -> Self {
        self.ops.push(Op::PadY(pixels));
        self
    }
    /// Pad both ends of whichever dimension is currently larger.
    pub fn pad_longest_side(mut self, pixels: u32) -> Self {
        self.ops.push(Op::PadLongestSide(pixels));
        self
    }
    /// Pad both ends of whichever dimension is currently smaller.
    pub fn pad_shortest_side(mut self, pixels: u32) -> Self {
        self.ops.push(Op::PadShortestSide(pixels));
        self
    }

    // ── scale / resize ────────────────────────────────────────────────────────

    /// Scale both dimensions by `factor` (e.g. `0.5` halves the image).
    pub fn scale(mut self, factor: f32) -> Self {
        self.ops.push(Op::Scale(factor));
        self
    }

    /// Resize to an explicit size. Pass `None` for a dimension to infer it
    /// from the other while preserving the aspect ratio.
    pub fn resize(mut self, width: Option<u32>, height: Option<u32>) -> Self {
        assert!(
            width.is_some() || height.is_some(),
            "resize: at least one of width / height must be Some"
        );
        self.ops.push(Op::Resize(width, height));
        self
    }

    /// Resize so the longest side equals `pixels`, preserving aspect ratio.
    pub fn resize_longest_side(mut self, pixels: u32) -> Self {
        self.ops.push(Op::ResizeLongestSide(pixels));
        self
    }

    /// Resize so the shortest side equals `pixels`, preserving aspect ratio.
    pub fn resize_shortest_side(mut self, pixels: u32) -> Self {
        self.ops.push(Op::ResizeShortestSide(pixels));
        self
    }

    // ── terminal ──────────────────────────────────────────────────────────────

    /// Apply all queued operations and return the processed image.
    pub fn process(self) -> DynamicImage {
        let pad_color = self.pad_color;
        let mut img = self.image;

        for op in self.ops {
            img = apply_op(img, op, pad_color);
        }

        img
    }

    /// Alias for [`process`](Self::process).
    #[inline]
    pub fn get_image(self) -> DynamicImage {
        self.process()
    }
}

// ── Op dispatch ──────────────────────────────────────────────────────────────

fn apply_op(img: DynamicImage, op: Op, pad_color: Rgba<u8>) -> DynamicImage {
    let (w, h) = (img.width(), img.height());

    match op {
        // ── crop ──────────────────────────────────────────────────────────────
        Op::CropTop(n) => {
            let n = n.min(h);
            img.crop_imm(0, n, w, h - n)
        }
        Op::CropBottom(n) => {
            let n = n.min(h);
            img.crop_imm(0, 0, w, h - n)
        }
        Op::CropLeft(n) => {
            let n = n.min(w);
            img.crop_imm(n, 0, w - n, h)
        }
        Op::CropRight(n) => {
            let n = n.min(w);
            img.crop_imm(0, 0, w - n, h)
        }
        Op::CropX(n) => {
            let each = (n * 2).min(w) / 2; // clamp so we don't go negative
            img.crop_imm(each, 0, w.saturating_sub(each * 2), h)
        }
        Op::CropY(n) => {
            let each = (n * 2).min(h) / 2;
            img.crop_imm(0, each, w, h.saturating_sub(each * 2))
        }
        Op::CropLongestSide(n) => {
            if w >= h {
                apply_op(img, Op::CropX(n), pad_color)
            } else {
                apply_op(img, Op::CropY(n), pad_color)
            }
        }
        Op::CropShortestSide(n) => {
            if w <= h {
                apply_op(img, Op::CropX(n), pad_color)
            } else {
                apply_op(img, Op::CropY(n), pad_color)
            }
        }

        // ── pad ───────────────────────────────────────────────────────────────
        Op::PadTop(n) => pad_image(img, 0, n, 0, 0, pad_color),
        Op::PadBottom(n) => pad_image(img, 0, 0, 0, n, pad_color),
        Op::PadLeft(n) => pad_image(img, n, 0, 0, 0, pad_color),
        Op::PadRight(n) => pad_image(img, 0, 0, n, 0, pad_color),
        Op::PadX(n) => pad_image(img, n, 0, n, 0, pad_color),
        Op::PadY(n) => pad_image(img, 0, n, 0, n, pad_color),
        Op::PadLongestSide(n) => {
            if w >= h {
                apply_op(img, Op::PadX(n), pad_color)
            } else {
                apply_op(img, Op::PadY(n), pad_color)
            }
        }
        Op::PadShortestSide(n) => {
            if w <= h {
                apply_op(img, Op::PadX(n), pad_color)
            } else {
                apply_op(img, Op::PadY(n), pad_color)
            }
        }

        // ── scale / resize ────────────────────────────────────────────────────
        Op::Scale(factor) => {
            assert!(factor > 0.0, "scale factor must be positive");
            let nw = ((w as f32 * factor).round() as u32).max(1);
            let nh = ((h as f32 * factor).round() as u32).max(1);
            img.resize_exact(nw, nh, imageops::FilterType::Lanczos3)
        }
        Op::Resize(target_w, target_h) => {
            let (nw, nh) = resolve_resize(w, h, target_w, target_h);
            img.resize_exact(nw, nh, imageops::FilterType::Lanczos3)
        }
        Op::ResizeLongestSide(pixels) => {
            let (target_w, target_h) = if w >= h {
                (Some(pixels), None)
            } else {
                (None, Some(pixels))
            };
            let (nw, nh) = resolve_resize(w, h, target_w, target_h);
            img.resize_exact(nw, nh, imageops::FilterType::Lanczos3)
        }
        Op::ResizeShortestSide(pixels) => {
            let (target_w, target_h) = if w <= h {
                (Some(pixels), None)
            } else {
                (None, Some(pixels))
            };
            let (nw, nh) = resolve_resize(w, h, target_w, target_h);
            img.resize_exact(nw, nh, imageops::FilterType::Lanczos3)
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Add a solid-colour border around `img`.
/// Arguments are (left, top, right, bottom) pixel counts.
fn pad_image(
    img: DynamicImage,
    left: u32,
    top: u32,
    right: u32,
    bottom: u32,
    color: Rgba<u8>,
) -> DynamicImage {
    let (w, h) = (img.width(), img.height());
    let new_w = w + left + right;
    let new_h = h + top + bottom;

    let mut canvas = RgbaImage::from_pixel(new_w, new_h, color);
    // overlay consumes a view of img; convert to rgba8 for compatibility
    imageops::overlay(&mut canvas, &img.to_rgba8(), left, top);
    DynamicImage::ImageRgba8(canvas)
}

/// Given current `(w, h)` and optional target dimensions, return the concrete
/// `(new_w, new_h)` — inferring the missing axis via aspect-ratio arithmetic.
fn resolve_resize(w: u32, h: u32, target_w: Option<u32>, target_h: Option<u32>) -> (u32, u32) {
    match (target_w, target_h) {
        (Some(nw), Some(nh)) => (nw.max(1), nh.max(1)),
        (Some(nw), None) => {
            let nh = ((nw as f64 / w as f64) * h as f64).round() as u32;
            (nw.max(1), nh.max(1))
        }
        (None, Some(nh)) => {
            let nw = ((nh as f64 / h as f64) * w as f64).round() as u32;
            (nw.max(1), nh.max(1))
        }
        (None, None) => unreachable!("validated in resize()"),
    }
}

// ── Example usage (remove or gate behind #[cfg(test)]) ───────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use image::DynamicImage;

    fn blank(w: u32, h: u32) -> DynamicImage {
        DynamicImage::ImageRgba8(RgbaImage::new(w, h))
    }

    #[test]
    fn chain_example() {
        let result = ImageProcessor::builder()
            .image(blank(200, 100))
            .pad_color(Rgba([255, 255, 255, 255]))
            .build()
            .crop_x(20) // 200 → 160
            .pad_y(10) // 100 → 120
            .scale(0.7) // 160×120 → 112×84
            .get_image();

        assert_eq!(result.width(), 112);
        assert_eq!(result.height(), 84);
    }

    #[test]
    fn resize_aspect_ratio() {
        let result = ImageProcessor::builder()
            .image(blank(400, 200))
            .build()
            .resize(Some(200), None) // height inferred → 100
            .get_image();

        assert_eq!(result.width(), 200);
        assert_eq!(result.height(), 100);
    }

    #[test]
    fn resize_longest_side_landscape() {
        // 400×200 — longest side is width
        let result = ImageProcessor::builder()
            .image(blank(400, 200))
            .build()
            .resize_longest_side(200) // width 400→200, height 200→100
            .get_image();

        assert_eq!(result.width(), 200);
        assert_eq!(result.height(), 100);
    }

    #[test]
    fn resize_longest_side_portrait() {
        // 200×400 — longest side is height
        let result = ImageProcessor::builder()
            .image(blank(200, 400))
            .build()
            .resize_longest_side(200) // height 400→200, width 200→100
            .get_image();

        assert_eq!(result.width(), 100);
        assert_eq!(result.height(), 200);
    }

    #[test]
    fn resize_shortest_side_landscape() {
        // 400×200 — shortest side is height
        let result = ImageProcessor::builder()
            .image(blank(400, 200))
            .build()
            .resize_shortest_side(100) // height 200→100, width 400→200
            .get_image();

        assert_eq!(result.width(), 200);
        assert_eq!(result.height(), 100);
    }
}
