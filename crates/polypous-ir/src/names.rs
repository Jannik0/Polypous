//! Naming rules, `docs/specs/graph-ir.md` §2.1 and §2.3.1.

use crate::limits;

fn is_name_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.')
}

/// Node ids and graph input, output and state names: 1–128 bytes of ASCII letters, digits, `_`,
/// `-` and `.`.
pub fn is_value_name(s: &str) -> bool {
    !s.is_empty() && s.len() <= limits::MAX_VALUE_NAME_BYTES && s.bytes().all(is_name_byte)
}

/// Parameter names: 1–256 bytes of the same characters as value names.
pub fn is_param_name(s: &str) -> bool {
    !s.is_empty() && s.len() <= limits::MAX_PARAM_NAME_BYTES && s.bytes().all(is_name_byte)
}

/// A `params_prefix` or `state_prefix` as written: up to 256 bytes of name characters and `{i}`
/// placeholders (possibly empty). Returns the number of `{i}` placeholders, or `None` if invalid.
pub fn prefix_placeholders(s: &str) -> Option<usize> {
    if s.len() > limits::MAX_PARAM_NAME_BYTES {
        return None;
    }
    let bytes = s.as_bytes();
    let mut count = 0;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'{' {
            if bytes.get(i + 1) == Some(&b'i') && bytes.get(i + 2) == Some(&b'}') {
                count += 1;
                i += 3;
                continue;
            }
            return None;
        }
        if !is_name_byte(bytes[i]) {
            return None;
        }
        i += 1;
    }
    Some(count)
}

/// Replaces every `{i}` in a prefix by `index`.
pub fn expand_prefix(prefix: &str, index: u64) -> String {
    prefix.replace("{i}", &index.to_string())
}

/// Dim symbols: `(letter | "_") { letter | digit | "_" }`, 1–64 bytes, not `min` or `max`.
pub fn is_symbol(s: &str) -> bool {
    let bytes = s.as_bytes();
    let Some(&first) = bytes.first() else {
        return false;
    };
    s.len() <= limits::MAX_SYMBOL_BYTES
        && (first.is_ascii_alphabetic() || first == b'_')
        && bytes
            .iter()
            .all(|&b| b.is_ascii_alphanumeric() || b == b'_')
        && s != "min"
        && s != "max"
}

/// Number of decimal digits of `n`.
pub fn decimal_digits(mut n: u64) -> usize {
    let mut d = 1;
    while n >= 10 {
        n /= 10;
        d += 1;
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_names() {
        assert!(is_value_name("fc1"));
        assert!(is_value_name("a.b-c_D9"));
        assert!(!is_value_name(""));
        assert!(!is_value_name("a:b"));
        assert!(!is_value_name("a b"));
        assert!(!is_value_name("ü"));
        assert!(is_value_name(&"a".repeat(128)));
        assert!(!is_value_name(&"a".repeat(129)));
        assert!(is_param_name(&"a".repeat(256)));
        assert!(!is_param_name(&"a".repeat(257)));
        assert!(!is_param_name("layers.{i}.w"));
    }

    #[test]
    fn prefixes() {
        assert_eq!(prefix_placeholders("layers.{i}."), Some(1));
        assert_eq!(prefix_placeholders(""), Some(0));
        assert_eq!(prefix_placeholders("{i}{i}"), Some(2));
        assert_eq!(prefix_placeholders("{j}"), None);
        assert_eq!(prefix_placeholders("{i"), None);
        assert_eq!(prefix_placeholders("a:b"), None);
        assert_eq!(expand_prefix("l.{i}.{i}", 12), "l.12.12");
    }

    #[test]
    fn symbols() {
        assert!(is_symbol("batch"));
        assert!(is_symbol("_x1"));
        assert!(!is_symbol("1x"));
        assert!(!is_symbol("min"));
        assert!(!is_symbol("max"));
        assert!(!is_symbol("a.b"));
        assert!(!is_symbol(""));
        assert!(is_symbol(&"s".repeat(64)));
        assert!(!is_symbol(&"s".repeat(65)));
    }

    #[test]
    fn digits() {
        assert_eq!(decimal_digits(0), 1);
        assert_eq!(decimal_digits(9), 1);
        assert_eq!(decimal_digits(10), 2);
        assert_eq!(decimal_digits(65_535), 5);
    }
}
