//! The text the model reads for a state, an instruction or an option, exactly as training rendered it
//! (`typestrict/format.py` `render()`, ported by the playground's `render.ts`):
//!
//! - a string is stripped like Python's `str.strip()`;
//! - anything else becomes `json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))`.
//!
//! One token of difference changes the answer and its calibration without raising any error, so this must match the
//! Python byte for byte. `serde_json` gets there with two conditions: its `Map` sorts keys (the crate is built without
//! `preserve_order`; `render_sorts_keys_by_code_point` breaks if a dependency turns it on), and floats are written by
//! [`PyFloat`], because `serde_json` writes `1e-5` as `0.00001` and `1e-7` as `1e-7` where Python writes `1e-05` and
//! `1e-07`.
//!
//! Where JSON numbers are concerned this follows Python, not JavaScript: `2.0` in the request stays a float and renders
//! as `2.0` (the playground, which has no int/float split, writes `2`).

use std::io;

use serde_json::Value;
use serde_json::ser::Formatter;

/// Content of a request: `state`, `instructions` and each criterion.
pub type Content = Value;

/// The rendered text; nothing (`None` or `null`) renders as an empty string.
pub fn render(content: Option<&Content>) -> String {
    match content {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => py_strip(s).to_string(),
        Some(v) => {
            let mut buf = Vec::new();
            let mut ser = serde_json::Serializer::with_formatter(&mut buf, PyFloat);
            serde::Serialize::serialize(v, &mut ser).expect("a Value always serializes");
            String::from_utf8(buf).expect("serde_json writes UTF-8")
        }
    }
}

/// `str.isspace()`: Unicode whitespace plus the separators U+001C..U+001F, which Rust's `char::is_whitespace` leaves out.
fn py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// Python's `str.strip()`.
pub fn py_strip(s: &str) -> &str {
    s.trim_matches(py_space)
}

/// A finite or non-finite `f64` as Python's `repr()` writes it (which is what `json.dumps` writes).
pub fn repr_f64(v: f64) -> String {
    if v.is_nan() {
        return "NaN".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    // `{:e}` gives the shortest round-trip mantissa with its exponent: "1e-7", "3.14e0", "1e21".
    let sci = format!("{v:e}");
    let (mantissa, exp) = sci.split_once('e').expect("{:e} always has an exponent");
    let exp: i32 = exp.parse().expect("the exponent is an integer");
    if (-4..16).contains(&exp) {
        // Fixed notation; Rust's Display never goes scientific, but drops the ".0" Python always writes.
        let mut s = format!("{v}");
        if !s.contains('.') {
            s.push_str(".0");
        }
        s
    } else {
        format!("{mantissa}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
    }
}

/// `serde_json` formatter that writes floats as Python does.
#[derive(Clone, Debug, Default)]
pub struct PyFloat;

impl Formatter for PyFloat {
    fn write_f64<W: ?Sized + io::Write>(&mut self, w: &mut W, v: f64) -> io::Result<()> {
        w.write_all(repr_f64(v).as_bytes())
    }

    fn write_f32<W: ?Sized + io::Write>(&mut self, w: &mut W, v: f32) -> io::Result<()> {
        w.write_all(repr_f64(f64::from(v)).as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn r(text: &str) -> String {
        render(Some(&serde_json::from_str(text).unwrap()))
    }

    #[test]
    fn nothing_renders_empty() {
        assert_eq!(render(None), "");
        assert_eq!(render(Some(&Value::Null)), "");
    }

    #[test]
    fn strings_are_stripped_like_python() {
        assert_eq!(render(Some(&json!("  \t\n hello  \u{3000}"))), "hello");
        assert_eq!(render(Some(&json!("  x \u{85} "))), "x");
        assert_eq!(render(Some(&json!("\u{1c}x\u{1f}"))), "x");
        assert_eq!(render(Some(&json!("   "))), "");
    }

    #[test]
    fn scalars_render_as_json() {
        assert_eq!(r("true"), "true");
        assert_eq!(r("false"), "false");
        assert_eq!(r("42"), "42");
        assert_eq!(r("[null, -7]"), "[null,-7]");
    }

    #[test]
    fn floats_print_like_python_repr() {
        assert_eq!(r("0.5"), "0.5");
        assert_eq!(r("-2.25"), "-2.25");
        assert_eq!(r("123.456"), "123.456");
        assert_eq!(r("0.0001"), "0.0001");
        assert_eq!(r("0.00001234"), "1.234e-05");
        assert_eq!(r("1.5e-16"), "1.5e-16");
        assert_eq!(r("1e-7"), "1e-07");
        assert_eq!(r("12345678.9"), "12345678.9");
        // A float stays a float, as in Python (JavaScript would print 2 and 15000000000000000).
        assert_eq!(r("2.0"), "2.0");
        assert_eq!(r("1.5e16"), "1.5e+16");
        assert_eq!(repr_f64(-0.0), "-0.0");
        assert_eq!(repr_f64(f64::MAX), "1.7976931348623157e+308");
    }

    #[test]
    fn strings_inside_json_escape_like_ensure_ascii_false() {
        assert_eq!(r(r#"["a\"b\\c\n\r\t\b\f\u0001é</>& "]"#), "[\"a\\\"b\\\\c\\n\\r\\t\\b\\f\\u0001é</>&\u{2028}\"]");
    }

    #[test]
    fn render_sorts_keys_by_code_point() {
        assert_eq!(
            r(r#"{"b": 1, "a": {"d": [1, "x"], "c": null}, "😀": 1, "！": 2, "ab": 3}"#),
            r#"{"a":{"c":null,"d":[1,"x"]},"ab":3,"b":1,"！":2,"😀":1}"#
        );
    }
}
