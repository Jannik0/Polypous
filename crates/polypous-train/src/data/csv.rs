//! The `csv` dataset (`docs/specs/training.md` §3.5.3): Polypous' own RFC 4180 parser.
//!
//! The rules, exactly as §3.5.3 writes them: records are separated by LF or CRLF, fields by `,`; a
//! field may be enclosed in `"`, with `""` for a quote inside, and may then contain `,` and line
//! breaks; there are no other escapes, no comments and no whitespace trimming. The first record is
//! the header. A UTF-8 BOM at the start is skipped, and an empty last line is allowed (the text
//! after the final line break is empty; it is not a record). Following RFC 4180, a `"` inside an
//! unquoted field and anything but `,` or a line break after a closing quote are malformed. A CR
//! that isn't followed by LF is part of the field (it is not a separator).
//!
//! The file is streamed: at most one record (1 MiB) is held besides the decoded fields, which
//! grow with fallible reservations and count against `data.max_bytes` record by record.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::num::IntErrorKind;
use std::path::Path;

use polypous_backend::{DType, TensorData};

use super::config::{CsvColumnConfig, CsvDtype, MAX_CSV_COLUMNS};
use super::{Budget, DataError, Dataset, FieldSelection, MAX_SAMPLES};

/// The largest CSV file: 4 GiB (§3.5.3).
pub(crate) const MAX_FILE_BYTES: u64 = 1 << 32;
/// The largest record, in bytes, without its line break: 1 MiB (§3.5.3).
pub(crate) const MAX_RECORD_BYTES: usize = 1 << 20;

const BOM: [u8; 3] = [0xEF, 0xBB, 0xBF];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    /// At the start of a field.
    FieldStart,
    /// Inside an unquoted field.
    Unquoted,
    /// Inside a quoted field.
    Quoted,
    /// After a `"` inside a quoted field: `""` or the closing quote.
    QuoteInQuoted,
    /// After a CR outside quotes; `after_quote` if it came right after a closing quote.
    Cr { after_quote: bool },
}

/// A push parser: [`Parser::push`] one byte at a time, [`Parser::finish`] at the end of the file.
struct Parser {
    state: State,
    /// The current record's field contents, unescaped and concatenated.
    data: Vec<u8>,
    /// The end offset in `data` of each finished field of the current record.
    ends: Vec<usize>,
    /// The record's bytes so far, without line breaks that end it.
    raw: usize,
    /// Whether the current record has any byte (so an empty remainder at the end isn't one).
    started: bool,
}

/// Why a record is malformed, before the record number and column are added.
struct Malformed {
    /// The 0-based field the problem is in.
    field: usize,
    message: String,
}

impl Parser {
    fn new() -> Self {
        Self {
            state: State::FieldStart,
            data: Vec::new(),
            ends: Vec::new(),
            raw: 0,
            started: false,
        }
    }

    fn clear(&mut self) {
        self.state = State::FieldStart;
        self.data.clear();
        self.ends.clear();
        self.raw = 0;
        self.started = false;
    }

    fn field(&self, i: usize) -> &[u8] {
        let start = if i == 0 { 0 } else { self.ends[i - 1] };
        &self.data[start..self.ends[i]]
    }

    fn malformed(&self, message: impl Into<String>) -> Malformed {
        Malformed {
            field: self.ends.len(),
            message: message.into(),
        }
    }

    /// Adds a content byte of the record (not a line break that ends it).
    fn content(&mut self, b: Option<u8>) -> Result<(), Malformed> {
        self.raw += 1;
        if self.raw > MAX_RECORD_BYTES {
            return Err(self.malformed("the record is longer than 1 MiB"));
        }
        if let Some(b) = b {
            // The record buffer is bounded by MAX_RECORD_BYTES, reserved fallibly.
            if self.data.len() == self.data.capacity() && self.data.try_reserve(1).is_err() {
                return Err(self.malformed("out of memory for the record"));
            }
            self.data.push(b);
        }
        Ok(())
    }

    fn end_field(&mut self) -> Result<(), Malformed> {
        if self.ends.len() >= MAX_CSV_COLUMNS {
            return Err(
                self.malformed(format!("the record has more than {MAX_CSV_COLUMNS} fields"))
            );
        }
        if self.ends.len() == self.ends.capacity() && self.ends.try_reserve(1).is_err() {
            return Err(self.malformed("out of memory for the record"));
        }
        self.ends.push(self.data.len());
        Ok(())
    }

    /// Processes one byte. Returns `true` when it ended a record.
    fn push(&mut self, b: u8) -> Result<bool, Malformed> {
        self.started = true;
        match self.state {
            State::FieldStart | State::Unquoted => match b {
                b',' => {
                    self.content(None)?;
                    self.end_field()?;
                    self.state = State::FieldStart;
                }
                b'\n' => {
                    self.end_field()?;
                    return Ok(true);
                }
                b'\r' => self.state = State::Cr { after_quote: false },
                b'"' if self.state == State::FieldStart => {
                    self.content(None)?;
                    self.state = State::Quoted;
                }
                b'"' => {
                    return Err(self.malformed(
                        "a `\"` inside an unquoted field (enclose the field in quotes and write \
                         `\"\"`)",
                    ));
                }
                _ => {
                    self.content(Some(b))?;
                    self.state = State::Unquoted;
                }
            },
            State::Quoted => {
                if b == b'"' {
                    self.content(None)?;
                    self.state = State::QuoteInQuoted;
                } else {
                    self.content(Some(b))?;
                }
            }
            State::QuoteInQuoted => match b {
                b'"' => {
                    self.content(Some(b'"'))?;
                    self.state = State::Quoted;
                }
                b',' => {
                    self.content(None)?;
                    self.end_field()?;
                    self.state = State::FieldStart;
                }
                b'\n' => {
                    self.end_field()?;
                    return Ok(true);
                }
                b'\r' => self.state = State::Cr { after_quote: true },
                _ => {
                    return Err(self.malformed(
                        "a character after the closing quote of a field (expected `,` or a line \
                         break)",
                    ));
                }
            },
            State::Cr { after_quote } => {
                if b == b'\n' {
                    self.end_field()?;
                    return Ok(true);
                }
                if after_quote {
                    return Err(self.malformed(
                        "a CR after the closing quote of a field that isn't followed by LF",
                    ));
                }
                // A lone CR is field content; reprocess `b` in an unquoted field.
                self.content(Some(b'\r'))?;
                self.state = State::Unquoted;
                return self.push(b);
            }
        }
        Ok(false)
    }

    /// At the end of the file. Returns `true` if a last record (without a line break) ended.
    fn finish(&mut self) -> Result<bool, Malformed> {
        if !self.started {
            return Ok(false);
        }
        match self.state {
            State::Quoted => Err(self.malformed("an unterminated quoted field")),
            State::Cr { after_quote: true } => {
                Err(self
                    .malformed("a CR after the closing quote of a field that isn't followed by LF"))
            }
            State::Cr { after_quote: false } => {
                self.content(Some(b'\r'))?;
                self.end_field()?;
                Ok(true)
            }
            _ => {
                self.end_field()?;
                Ok(true)
            }
        }
    }
}

/// A used field: its name, columns (header indexes), dtype and data.
struct Column {
    name: String,
    indexes: Vec<usize>,
    dtype: CsvDtype,
    data: Vec<u8>,
}

impl Column {
    fn element_bytes(&self) -> usize {
        match self.dtype {
            CsvDtype::F32 | CsvDtype::I32 => 4,
            CsvDtype::I64 => 8,
        }
    }
}

/// A bounded preview of a value for an error message.
fn preview(value: &str) -> String {
    const MAX: usize = 64;
    if value.len() <= MAX {
        return format!("{value:?}");
    }
    let mut end = MAX;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    format!("{:?}…", &value[..end])
}

/// Parses one value (§3.5.3) into `out`, little-endian.
fn parse_value(value: &str, dtype: CsvDtype, out: &mut Vec<u8>) -> Result<(), String> {
    if value.is_empty() {
        return Err("empty value".to_owned());
    }
    match dtype {
        CsvDtype::F32 => {
            let v: f32 = value
                .parse()
                .map_err(|_| format!("{} is not an f32 number", preview(value)))?;
            // `f32::from_str` also accepts `inf`, `infinity` and `nan` in any case, and rounds
            // values beyond f32's range to infinity: all of these are errors (§3.5.3).
            if !v.is_finite() {
                return Err(format!(
                    "{} is not a finite f32 (inf, NaN and values that round to infinity are not \
                     allowed)",
                    preview(value)
                ));
            }
            out.extend_from_slice(&v.to_le_bytes());
        }
        CsvDtype::I64 => {
            let v: i64 = value.parse().map_err(|e| int_error(value, &e, "i64"))?;
            out.extend_from_slice(&v.to_le_bytes());
        }
        CsvDtype::I32 => {
            let v: i32 = value.parse().map_err(|e| int_error(value, &e, "i32"))?;
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    Ok(())
}

fn int_error(value: &str, e: &std::num::ParseIntError, dtype: &str) -> String {
    match e.kind() {
        IntErrorKind::PosOverflow | IntErrorKind::NegOverflow => {
            format!("{} is out of the {dtype} range", preview(value))
        }
        _ => format!("{} is not a decimal integer", preview(value)),
    }
}

/// An error at record `record` (1 = the header), column `column`.
fn at(record: u64, column: Option<&str>, message: &str) -> DataError {
    DataError::invalid(match column {
        Some(c) => format!("record {record}, column `{c}`: {message}"),
        None => format!("record {record}: {message}"),
    })
    .with_record(record, column)
}

pub(crate) fn read(
    path: &Path,
    columns: &BTreeMap<String, CsvColumnConfig>,
    fields: &FieldSelection,
    budget: &mut Budget,
) -> Result<Dataset, DataError> {
    read_inner(path, columns, fields, budget).map_err(|e| e.with_file(path))
}

fn read_inner(
    path: &Path,
    columns: &BTreeMap<String, CsvColumnConfig>,
    fields: &FieldSelection,
    budget: &mut Budget,
) -> Result<Dataset, DataError> {
    let len = std::fs::metadata(path)
        .map_err(|e| DataError::io(e, path, "read the CSV file"))?
        .len();
    if len > MAX_FILE_BYTES {
        return Err(DataError::invalid(format!(
            "the CSV file is {len} bytes; at most 4 GiB ({MAX_FILE_BYTES} bytes) are allowed"
        )));
    }
    let file = File::open(path).map_err(|e| DataError::io(e, path, "open the CSV file"))?;
    // At most one byte past the limit is read, so a file that grows meanwhile is caught.
    let mut reader = BufReader::with_capacity(64 * 1024, file.take(MAX_FILE_BYTES + 1));
    let mut total = 0u64;
    let mut parser = Parser::new();
    let mut record = 1u64;
    let mut header: Option<Vec<String>> = None;
    let mut used: Vec<Column> = Vec::new();
    let mut samples = 0u64;
    let mut bom = 0usize;
    let mut at_start = true;
    loop {
        let chunk = reader
            .fill_buf()
            .map_err(|e| DataError::io(e, path, "read the CSV file"))?;
        if chunk.is_empty() {
            break;
        }
        let n = chunk.len();
        total += n as u64;
        if total > MAX_FILE_BYTES {
            return Err(DataError::invalid(
                "the CSV file grew past 4 GiB while it was read",
            ));
        }
        let mut i = 0;
        // The BOM may straddle chunks: match it byte by byte at the start of the file.
        while at_start && i < n {
            if chunk[i] == BOM[bom] {
                bom += 1;
                i += 1;
                if bom == BOM.len() {
                    at_start = false;
                }
            } else {
                at_start = false;
                // A partial BOM is content: feed the bytes it matched.
                for &b in &BOM[..bom] {
                    if parser
                        .push(b)
                        .map_err(|m| malformed(&m, record, header.as_deref()))?
                    {
                        unreachable_record()?;
                    }
                }
            }
        }
        for &b in &chunk[i..] {
            let ended = parser
                .push(b)
                .map_err(|m| malformed(&m, record, header.as_deref()))?;
            if ended {
                on_record(
                    &parser,
                    record,
                    &mut header,
                    &mut used,
                    columns,
                    fields,
                    budget,
                    &mut samples,
                )?;
                parser.clear();
                record += 1;
            }
        }
        reader.consume(n);
    }
    if at_start && bom > 0 {
        for &b in &BOM[..bom] {
            parser
                .push(b)
                .map_err(|m| malformed(&m, record, header.as_deref()))?;
        }
    }
    if parser
        .finish()
        .map_err(|m| malformed(&m, record, header.as_deref()))?
    {
        on_record(
            &parser,
            record,
            &mut header,
            &mut used,
            columns,
            fields,
            budget,
            &mut samples,
        )?;
    }
    if header.is_none() {
        return Err(DataError::invalid(
            "the CSV file is empty; it needs a header record",
        ));
    }
    if samples == 0 {
        return Err(DataError::invalid(
            "the CSV file has a header but no records; a dataset has at least one sample",
        ));
    }
    if used.is_empty() {
        return Err(DataError::invalid(format!(
            "none of the CSV dataset's fields is used by the training (it has {:?})",
            columns.keys().collect::<Vec<_>>()
        )));
    }
    let mut out = BTreeMap::new();
    for column in used {
        let mut shape = vec![samples];
        if let Some(CsvColumnConfig {
            from: super::ColumnSource::Many(names),
            ..
        }) = columns.get(&column.name)
        {
            shape.push(names.len() as u64);
        }
        let dtype = match column.dtype {
            CsvDtype::F32 => DType::F32,
            CsvDtype::I64 => DType::I64,
            CsvDtype::I32 => DType::I32,
        };
        out.insert(
            column.name,
            TensorData {
                dtype,
                shape,
                bytes: column.data,
            },
        );
    }
    Ok(Dataset::from_checked(samples, out))
}

/// The BOM bytes can't end a record (none is a line break).
fn unreachable_record() -> Result<(), DataError> {
    Err(DataError::invalid("internal: a BOM byte ended a record"))
}

fn malformed(m: &Malformed, record: u64, header: Option<&[String]>) -> DataError {
    let column = header.and_then(|h| h.get(m.field)).map(String::as_str);
    at(record, column, &m.message)
}

#[allow(
    clippy::too_many_arguments,
    reason = "the parser loop's state, passed by reference"
)]
fn on_record(
    parser: &Parser,
    record: u64,
    header: &mut Option<Vec<String>>,
    used: &mut Vec<Column>,
    columns: &BTreeMap<String, CsvColumnConfig>,
    fields: &FieldSelection,
    budget: &mut Budget,
    samples: &mut u64,
) -> Result<(), DataError> {
    let count = parser.ends.len();
    let Some(names) = header.as_ref() else {
        let names = read_header(parser, record)?;
        *used = resolve(&names, columns, fields)?;
        *header = Some(names);
        return Ok(());
    };
    if count != names.len() {
        return Err(at(
            record,
            None,
            &format!(
                "the record has {count} fields, but the header has {}",
                names.len()
            ),
        ));
    }
    // UTF-8 for every field, used or not (the file must be UTF-8).
    let mut text = Vec::with_capacity(0);
    for (i, name) in names.iter().enumerate() {
        match std::str::from_utf8(parser.field(i)) {
            Ok(s) => text.push(s),
            Err(_) => {
                return Err(at(record, Some(name), "the field is not valid UTF-8"));
            }
        }
    }
    if *samples >= MAX_SAMPLES {
        return Err(at(
            record,
            None,
            "the file has more than 2^32 records; a dataset has at most 2^32 samples",
        ));
    }
    let bytes: u64 = used
        .iter()
        .map(|c| (c.indexes.len() * c.element_bytes()) as u64)
        .sum();
    budget
        .charge(bytes)
        .map_err(|e| e.with_record(record, None))?;
    for column in used.iter_mut() {
        let need = column.indexes.len() * column.element_bytes();
        column
            .data
            .try_reserve(need)
            .map_err(|_| DataError::out_of_memory(need as u64, "a CSV field"))?;
        for &i in &column.indexes {
            parse_value(text[i], column.dtype, &mut column.data)
                .map_err(|m| at(record, Some(&names[i]), &m).with_field(&column.name))?;
        }
    }
    *samples += 1;
    Ok(())
}

fn read_header(parser: &Parser, record: u64) -> Result<Vec<String>, DataError> {
    let count = parser.ends.len();
    if count > MAX_CSV_COLUMNS {
        return Err(at(
            record,
            None,
            &format!("the header has {count} columns; at most {MAX_CSV_COLUMNS} are allowed"),
        ));
    }
    let mut names = Vec::new();
    names
        .try_reserve_exact(count)
        .map_err(|_| DataError::out_of_memory(count as u64, "the CSV header"))?;
    let mut seen = BTreeSet::new();
    for i in 0..count {
        let Ok(name) = std::str::from_utf8(parser.field(i)) else {
            return Err(at(
                record,
                None,
                &format!("header column {} is not valid UTF-8", i + 1),
            ));
        };
        if name.is_empty() {
            return Err(at(
                record,
                None,
                &format!(
                    "header column {} is empty; column names must be non-empty",
                    i + 1
                ),
            ));
        }
        if !seen.insert(name) {
            return Err(at(
                record,
                Some(name),
                "the column name appears twice in the header; names must be unique",
            ));
        }
        names.push(name.to_owned());
    }
    Ok(names)
}

/// Maps every configured field's columns to header indexes (an unknown column is an error for
/// any field), and returns the used fields.
fn resolve(
    header: &[String],
    columns: &BTreeMap<String, CsvColumnConfig>,
    fields: &FieldSelection,
) -> Result<Vec<Column>, DataError> {
    let index: BTreeMap<&str, usize> = header
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();
    let mut used = Vec::new();
    for (name, column) in columns {
        let mut indexes = Vec::with_capacity(column.from.columns().len());
        for c in column.from.columns() {
            let Some(&i) = index.get(c.as_str()) else {
                return Err(DataError::invalid(format!(
                    "field `{name}`: the CSV header has no column `{c}`"
                ))
                .with_field(name));
            };
            indexes.push(i);
        }
        if fields.wants(name) {
            used.push(Column {
                name: name.clone(),
                indexes,
                dtype: column.dtype,
                data: Vec::new(),
            });
        }
    }
    Ok(used)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(input: &[u8]) -> Result<Vec<Vec<String>>, (u64, String)> {
        let mut p = Parser::new();
        let mut out = Vec::new();
        let mut record = 1;
        let take = |p: &Parser| {
            (0..p.ends.len())
                .map(|i| String::from_utf8_lossy(p.field(i)).into_owned())
                .collect::<Vec<_>>()
        };
        for &b in input {
            if p.push(b).map_err(|m| (record, m.message))? {
                out.push(take(&p));
                p.clear();
                record += 1;
            }
        }
        if p.finish().map_err(|m| (record, m.message))? {
            out.push(take(&p));
        }
        Ok(out)
    }

    #[test]
    fn rfc4180() {
        assert_eq!(parse(b"a,b\n1,2\n").unwrap(), [["a", "b"], ["1", "2"]]);
        assert_eq!(parse(b"a,b\r\n1,2").unwrap(), [["a", "b"], ["1", "2"]]);
        assert_eq!(
            parse(b"\"x,\"\"y\"\"\",\"line\r\nbreak\"\n").unwrap(),
            [["x,\"y\"", "line\r\nbreak"]]
        );
        assert_eq!(parse(b"a,\"\"\n").unwrap(), [["a", ""]]);
        assert_eq!(parse(b",\n").unwrap(), [["", ""]]);
        // An empty line in the middle is a record of one empty field; at the end it isn't one.
        assert_eq!(parse(b"a\n\nb\n").unwrap(), [["a"], [""], ["b"]]);
        assert_eq!(parse(b"a\n\n").unwrap(), [["a"], [""]]);
        assert_eq!(parse(b"").unwrap(), Vec::<Vec<String>>::new());
        // A lone CR is content.
        assert_eq!(parse(b"a\rb\n").unwrap(), [["a\rb"]]);
        assert_eq!(parse(b"a\r").unwrap(), [["a\r"]]);
        assert_eq!(parse(b"\"a\"\r\n").unwrap(), [["a"]]);
    }

    #[test]
    fn malformed_records() {
        assert!(parse(b"\"abc").unwrap_err().1.contains("unterminated"));
        assert!(parse(b"a\"b\n").unwrap_err().1.contains("unquoted"));
        assert!(parse(b"\"a\"b\n").unwrap_err().1.contains("closing quote"));
        assert!(parse(b"\"a\"\rb").unwrap_err().1.contains("CR"));
        assert!(parse(b"\"a\"\r").unwrap_err().1.contains("CR"));
        let (record, message) = parse(b"h\nok\n\"bad").unwrap_err();
        assert_eq!(record, 3);
        assert!(message.contains("unterminated"));
    }

    #[test]
    fn record_limit() {
        let mut long = vec![b'1'; MAX_RECORD_BYTES];
        long.push(b'\n');
        assert_eq!(parse(&long).unwrap()[0][0].len(), MAX_RECORD_BYTES);
        let mut too_long = vec![b'1'; MAX_RECORD_BYTES + 1];
        too_long.push(b'\n');
        assert!(parse(&too_long).unwrap_err().1.contains("1 MiB"));
        // The line break isn't counted; a CRLF neither.
        let mut crlf = vec![b'1'; MAX_RECORD_BYTES];
        crlf.extend_from_slice(b"\r\n");
        assert!(parse(&crlf).is_ok());
    }

    #[test]
    fn values() {
        let mut out = Vec::new();
        for ok in [
            "1",
            "-0.5",
            "1e-3",
            "+2",
            ".5",
            "5.",
            "1E3",
            "3.4028235e38",
            "1e-50",
        ] {
            out.clear();
            parse_value(ok, CsvDtype::F32, &mut out).unwrap();
            assert_eq!(out, ok.parse::<f32>().unwrap().to_le_bytes(), "{ok}");
        }
        for bad in [
            "", " 1", "1 ", "inf", "-inf", "Infinity", "NaN", "nan", "1e39", "-3.5e38", "0x10",
            "1,5", "--1", "e5",
        ] {
            assert!(
                parse_value(bad, CsvDtype::F32, &mut out).is_err(),
                "{bad:?}"
            );
        }
        out.clear();
        parse_value("-9223372036854775808", CsvDtype::I64, &mut out).unwrap();
        assert_eq!(out, i64::MIN.to_le_bytes());
        out.clear();
        parse_value("+7", CsvDtype::I32, &mut out).unwrap();
        assert_eq!(out, 7i32.to_le_bytes());
        assert!(
            parse_value("2147483648", CsvDtype::I32, &mut out)
                .unwrap_err()
                .contains("range")
        );
        assert!(
            parse_value("9223372036854775808", CsvDtype::I64, &mut out)
                .unwrap_err()
                .contains("range")
        );
        for bad in ["1.0", "1e3", " 1", "", "-", "0x1", "1_000"] {
            assert!(
                parse_value(bad, CsvDtype::I64, &mut out).is_err(),
                "{bad:?}"
            );
        }
    }
}
