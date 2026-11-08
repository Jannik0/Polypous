//! Symbolic dim expressions: grammar, limits, normal form, canonical string and equality,
//! `docs/specs/graph-ir.md` §2.3.1–§2.3.5.
//!
//! A [`DimExpr`] is always in normal form: a sum of terms `c * m` with `c ≠ 0`, sorted canonically.
//! Two dims are equal (§2.3.3) exactly when their `DimExpr` values are equal (`==`). Every operation
//! uses checked `i64` arithmetic and enforces the limits of §2.3.5, so building a normal form from
//! untrusted input never panics and stays small.

use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::fmt;

use crate::diag::ErrorCode;
use crate::limits;
use crate::names;

/// Why a dim expression is invalid or could not be normalized.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DimError {
    /// The string doesn't match the grammar of §2.3.1.
    #[error("dim expression `{expr}` doesn't parse: {reason}")]
    Syntax {
        /// The expression (truncated if long).
        expr: String,
        /// What went wrong.
        reason: String,
    },
    /// The string is longer than 256 bytes.
    #[error("dim expression is {len} bytes, more than the limit of {max}")]
    TooLong {
        /// Length in bytes.
        len: usize,
        /// The limit.
        max: usize,
    },
    /// Parentheses and `min`/`max` calls nest deeper than 16.
    #[error("dim expression nests deeper than {max} levels")]
    TooDeep {
        /// The limit.
        max: usize,
    },
    /// A symbol isn't declared in `dims`.
    #[error("dim symbol `{name}` is not declared in `dims`")]
    UndeclaredSymbol {
        /// The symbol.
        name: String,
    },
    /// The right operand of `//` or `%` isn't an integer literal ≥ 1 after normalization.
    #[error("the divisor of `{op}` must normalize to an integer literal >= 1, got `{divisor}`")]
    InvalidDivisor {
        /// `//` or `%`.
        op: &'static str,
        /// Canonical string of the divisor.
        divisor: String,
    },
    /// A normal form has more than 64 terms.
    #[error("dim expression has more than {max} terms after normalization")]
    TooManyTerms {
        /// The limit.
        max: usize,
    },
    /// A monomial has degree more than 8.
    #[error("dim expression has a term of degree more than {max} after normalization")]
    DegreeTooHigh {
        /// The limit.
        max: usize,
    },
    /// `i64` overflow while normalizing or evaluating.
    #[error("integer overflow in dim expression")]
    Overflow,
}

impl DimError {
    /// The validation error code for this error (§4.2).
    pub fn code(&self) -> ErrorCode {
        match self {
            DimError::Overflow => ErrorCode::DimOverflow,
            _ => ErrorCode::InvalidDimExpr,
        }
    }
}

/// Why a dim could not be evaluated for given symbol values.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EvalError {
    /// A symbol has no value.
    #[error("dim symbol `{0}` has no value")]
    Unbound(String),
    /// `i64` overflow.
    #[error("integer overflow while evaluating a dim")]
    Overflow,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum AtomKind {
    Symbol(String),
    /// `(e) // k` with `k ≥ 2`; `e` is not a literal.
    FloorDiv(Box<DimExpr>, i64),
    /// `min(a, b)`, arguments sorted by canonical string.
    Min(Box<DimExpr>, Box<DimExpr>),
    /// `max(a, b)`, arguments sorted by canonical string.
    Max(Box<DimExpr>, Box<DimExpr>),
}

/// A symbol or an opaque term. `key` is its canonical string, which determines it uniquely.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Atom {
    key: String,
    kind: AtomKind,
}

impl Atom {
    fn new(kind: AtomKind) -> Atom {
        let key = match &kind {
            AtomKind::Symbol(s) => s.clone(),
            AtomKind::FloorDiv(e, k) => {
                if e.is_plain_atom() {
                    format!("{e} // {k}")
                } else {
                    format!("({e}) // {k}")
                }
            }
            AtomKind::Min(a, b) => format!("min({a}, {b})"),
            AtomKind::Max(a, b) => format!("max({a}, {b})"),
        };
        Atom { key, kind }
    }

    fn is_floor_div(&self) -> bool {
        matches!(self.kind, AtomKind::FloorDiv(..))
    }
}

/// A product of atoms, sorted by key. `key` is the atoms' keys joined with `*`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Monomial {
    atoms: Vec<Atom>,
    key: String,
}

impl Monomial {
    fn new(mut atoms: Vec<Atom>) -> Monomial {
        atoms.sort_by(|a, b| a.key.cmp(&b.key));
        let key = atoms
            .iter()
            .map(|a| a.key.as_str())
            .collect::<Vec<_>>()
            .join("*");
        Monomial { atoms, key }
    }

    fn one() -> Monomial {
        Monomial {
            atoms: Vec::new(),
            key: String::new(),
        }
    }

    fn degree(&self) -> usize {
        self.atoms.len()
    }

    fn canonical_cmp(&self, other: &Monomial) -> Ordering {
        other
            .degree()
            .cmp(&self.degree())
            .then_with(|| self.key.cmp(&other.key))
    }

    fn mul(&self, other: &Monomial) -> Result<Monomial, DimError> {
        let degree = self.degree() + other.degree();
        if degree > limits::MAX_DIM_DEGREE {
            return Err(DimError::DegreeTooHigh {
                max: limits::MAX_DIM_DEGREE,
            });
        }
        let mut atoms = Vec::with_capacity(degree);
        atoms.extend(self.atoms.iter().cloned());
        atoms.extend(other.atoms.iter().cloned());
        Ok(Monomial::new(atoms))
    }

    /// Writes the monomial as a factor of a product (opaque `//` atoms in parentheses).
    fn write_product(&self, f: &mut fmt::Formatter<'_>, standalone: bool) -> fmt::Result {
        for (i, atom) in self.atoms.iter().enumerate() {
            if i > 0 {
                f.write_str("*")?;
            }
            if atom.is_floor_div() && !standalone {
                write!(f, "({})", atom.key)?;
            } else {
                f.write_str(&atom.key)?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Term {
    coeff: i64,
    mono: Monomial,
}

/// A dim in normal form (§2.3.2).
///
/// `Display` writes the canonical string, which parses back to the same normal form.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DimExpr {
    /// Sorted by degree (descending), then monomial key; the constant term, if any, is last.
    terms: Vec<Term>,
}

impl DimExpr {
    /// The literal `v`.
    pub fn literal(v: i64) -> Result<DimExpr, DimError> {
        if v == i64::MIN {
            // Excluded so that every coefficient has an `i64` absolute value (canonical strings
            // write `0 - |c|` and must parse back).
            return Err(DimError::Overflow);
        }
        Ok(DimExpr {
            terms: if v == 0 {
                Vec::new()
            } else {
                vec![Term {
                    coeff: v,
                    mono: Monomial::one(),
                }]
            },
        })
    }

    /// The plain symbol `name`. The caller is responsible for the name being a valid, declared
    /// symbol ([`names::is_symbol`]).
    pub fn symbol(name: &str) -> DimExpr {
        DimExpr {
            terms: vec![Term {
                coeff: 1,
                mono: Monomial::new(vec![Atom::new(AtomKind::Symbol(name.to_owned()))]),
            }],
        }
    }

    /// Parses a dim expression string (§2.3.1) and normalizes it. `is_declared` says whether a
    /// symbol is declared in `dims`.
    pub fn parse(text: &str, is_declared: &dyn Fn(&str) -> bool) -> Result<DimExpr, DimError> {
        if text.len() > limits::MAX_DIM_EXPR_BYTES {
            return Err(DimError::TooLong {
                len: text.len(),
                max: limits::MAX_DIM_EXPR_BYTES,
            });
        }
        let tokens = tokenize(text)?;
        let mut parser = Parser {
            text,
            tokens,
            pos: 0,
            depth: 0,
            is_declared,
        };
        let expr = parser.expr()?;
        if parser.pos != parser.tokens.len() {
            return Err(parser.syntax("unexpected token after the end of the expression"));
        }
        Ok(expr)
    }

    /// The value if this dim is an integer literal.
    pub fn as_literal(&self) -> Option<i64> {
        match self.terms.as_slice() {
            [] => Some(0),
            [t] if t.mono.atoms.is_empty() => Some(t.coeff),
            _ => None,
        }
    }

    /// The symbol name if this dim is a plain symbol.
    pub fn as_symbol(&self) -> Option<&str> {
        match self.terms.as_slice() {
            [t] if t.coeff == 1 => match t.mono.atoms.as_slice() {
                [
                    Atom {
                        kind: AtomKind::Symbol(s),
                        ..
                    },
                ] => Some(s),
                _ => None,
            },
            _ => None,
        }
    }

    /// Every symbol used, including inside opaque terms.
    pub fn symbols(&self) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        self.collect_symbols(&mut out);
        out
    }

    fn collect_symbols(&self, out: &mut BTreeSet<String>) {
        for t in &self.terms {
            for a in &t.mono.atoms {
                match &a.kind {
                    AtomKind::Symbol(s) => {
                        out.insert(s.clone());
                    }
                    AtomKind::FloorDiv(e, _) => e.collect_symbols(out),
                    AtomKind::Min(x, y) | AtomKind::Max(x, y) => {
                        x.collect_symbols(out);
                        y.collect_symbols(out);
                    }
                }
            }
        }
    }

    /// Number of terms of the normal form.
    pub fn term_count(&self) -> usize {
        self.terms.len()
    }

    /// `self + other`.
    pub fn add(&self, other: &DimExpr) -> Result<DimExpr, DimError> {
        let mut terms = Vec::with_capacity(self.terms.len() + other.terms.len());
        terms.extend(self.terms.iter().cloned());
        terms.extend(other.terms.iter().cloned());
        collect(terms)
    }

    /// `-self`.
    fn neg(&self) -> Result<DimExpr, DimError> {
        let terms = self
            .terms
            .iter()
            .map(|t| {
                Ok(Term {
                    coeff: t.coeff.checked_neg().ok_or(DimError::Overflow)?,
                    mono: t.mono.clone(),
                })
            })
            .collect::<Result<Vec<_>, DimError>>()?;
        collect(terms)
    }

    /// `self - other`.
    pub fn sub(&self, other: &DimExpr) -> Result<DimExpr, DimError> {
        self.add(&other.neg()?)
    }

    /// `self * other`.
    pub fn mul(&self, other: &DimExpr) -> Result<DimExpr, DimError> {
        let mut terms = Vec::with_capacity(self.terms.len().saturating_mul(other.terms.len()));
        for a in &self.terms {
            for b in &other.terms {
                terms.push(Term {
                    coeff: a.coeff.checked_mul(b.coeff).ok_or(DimError::Overflow)?,
                    mono: a.mono.mul(&b.mono)?,
                });
            }
        }
        collect(terms)
    }

    /// `self // k` (floor division) for a literal divisor `k ≥ 1` (§2.3.2 rule 2).
    pub fn floor_div(&self, k: i64) -> Result<DimExpr, DimError> {
        if k < 1 {
            return Err(DimError::InvalidDivisor {
                op: "//",
                divisor: k.to_string(),
            });
        }
        if let Some(v) = self.as_literal() {
            return DimExpr::literal(v.div_euclid(k));
        }
        if k == 1 {
            return Ok(self.clone());
        }
        let mut q_terms = Vec::new();
        let mut r_terms = Vec::new();
        for t in &self.terms {
            // k ≥ 2, so div_euclid/rem_euclid can't overflow; for a positive divisor they are
            // floor division and the matching remainder in [0, k).
            let q = t.coeff.div_euclid(k);
            let r = t.coeff.rem_euclid(k);
            if q != 0 {
                q_terms.push(Term {
                    coeff: q,
                    mono: t.mono.clone(),
                });
            }
            if r != 0 {
                r_terms.push(Term {
                    coeff: r,
                    mono: t.mono.clone(),
                });
            }
        }
        let q = collect(q_terms)?;
        let r = collect(r_terms)?;
        let r_div = match r.as_literal() {
            Some(v) => DimExpr::literal(v.div_euclid(k))?,
            None => DimExpr::atom(AtomKind::FloorDiv(Box::new(r), k)),
        };
        q.add(&r_div)
    }

    /// `self % k` for a literal divisor `k ≥ 1`, normalized as `self - k * (self // k)`.
    pub fn rem(&self, k: i64) -> Result<DimExpr, DimError> {
        if k < 1 {
            return Err(DimError::InvalidDivisor {
                op: "%",
                divisor: k.to_string(),
            });
        }
        let q = self.floor_div(k)?;
        self.sub(&DimExpr::literal(k)?.mul(&q)?)
    }

    /// `min(self, other)` (§2.3.2 rule 4).
    pub fn min(&self, other: &DimExpr) -> Result<DimExpr, DimError> {
        self.min_max(other, true)
    }

    /// `max(self, other)` (§2.3.2 rule 4).
    pub fn max(&self, other: &DimExpr) -> Result<DimExpr, DimError> {
        self.min_max(other, false)
    }

    fn min_max(&self, other: &DimExpr, is_min: bool) -> Result<DimExpr, DimError> {
        if let Some(d) = self.sub(other)?.as_literal() {
            let self_smaller = d <= 0;
            return Ok(if self_smaller == is_min {
                self.clone()
            } else {
                other.clone()
            });
        }
        let (a, b) = {
            let (sa, sb) = (self.to_string(), other.to_string());
            if sa <= sb {
                (self.clone(), other.clone())
            } else {
                (other.clone(), self.clone())
            }
        };
        let kind = if is_min {
            AtomKind::Min(Box::new(a), Box::new(b))
        } else {
            AtomKind::Max(Box::new(a), Box::new(b))
        };
        Ok(DimExpr::atom(kind))
    }

    fn atom(kind: AtomKind) -> DimExpr {
        DimExpr {
            terms: vec![Term {
                coeff: 1,
                mono: Monomial::new(vec![Atom::new(kind)]),
            }],
        }
    }

    /// A single term with coefficient 1 and one atom that isn't `//`: written without parentheses
    /// as the dividend of `//`.
    fn is_plain_atom(&self) -> bool {
        matches!(self.terms.as_slice(), [t] if t.coeff == 1 && t.mono.atoms.len() == 1 && !t.mono.atoms[0].is_floor_div())
    }

    /// Evaluates the dim with checked `i64` arithmetic.
    pub fn eval(&self, value_of: &dyn Fn(&str) -> Option<i64>) -> Result<i64, EvalError> {
        let mut sum: i64 = 0;
        for t in &self.terms {
            let mut prod = t.coeff;
            for a in &t.mono.atoms {
                let v = match &a.kind {
                    AtomKind::Symbol(s) => {
                        value_of(s).ok_or_else(|| EvalError::Unbound(s.clone()))?
                    }
                    AtomKind::FloorDiv(e, k) => e.eval(value_of)?.div_euclid(*k),
                    AtomKind::Min(x, y) => x.eval(value_of)?.min(y.eval(value_of)?),
                    AtomKind::Max(x, y) => x.eval(value_of)?.max(y.eval(value_of)?),
                };
                prod = prod.checked_mul(v).ok_or(EvalError::Overflow)?;
            }
            sum = sum.checked_add(prod).ok_or(EvalError::Overflow)?;
        }
        Ok(sum)
    }
}

/// Collects equal monomials, drops zero coefficients, sorts canonically and checks the limits.
fn collect(mut terms: Vec<Term>) -> Result<DimExpr, DimError> {
    terms.sort_by(|a, b| a.mono.canonical_cmp(&b.mono));
    let mut out: Vec<Term> = Vec::with_capacity(terms.len());
    for t in terms {
        match out.last_mut() {
            Some(last) if last.mono == t.mono => {
                last.coeff = last.coeff.checked_add(t.coeff).ok_or(DimError::Overflow)?;
            }
            _ => out.push(t),
        }
    }
    out.retain(|t| t.coeff != 0);
    if out.iter().any(|t| t.coeff == i64::MIN) {
        return Err(DimError::Overflow);
    }
    if out.len() > limits::MAX_DIM_TERMS {
        return Err(DimError::TooManyTerms {
            max: limits::MAX_DIM_TERMS,
        });
    }
    Ok(DimExpr { terms: out })
}

impl fmt::Display for DimExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.terms.is_empty() {
            return f.write_str("0");
        }
        for (i, t) in self.terms.iter().enumerate() {
            let neg = t.coeff < 0;
            if i == 0 {
                if neg {
                    // There is no unary minus; `0 - x` keeps the string parseable.
                    f.write_str("0 - ")?;
                }
            } else {
                f.write_str(if neg { " - " } else { " + " })?;
            }
            let abs = t.coeff.unsigned_abs();
            if t.mono.atoms.is_empty() {
                write!(f, "{abs}")?;
            } else if abs == 1 {
                t.mono.write_product(f, t.mono.atoms.len() == 1)?;
            } else {
                write!(f, "{abs}*")?;
                t.mono.write_product(f, false)?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    Int(i64),
    Ident(String),
    Plus,
    Minus,
    Star,
    SlashSlash,
    Percent,
    LParen,
    RParen,
    Comma,
}

fn truncate(text: &str) -> String {
    // Limited before parsing to 256 bytes; truncating keeps messages short anyway.
    text.chars().take(64).collect()
}

fn tokenize(text: &str) -> Result<Vec<Token>, DimError> {
    let syntax = |reason: String| DimError::Syntax {
        expr: truncate(text),
        reason,
    };
    let bytes = text.as_bytes();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        match b {
            b' ' | b'\t' | b'\n' | b'\r' => i += 1,
            b'+' => {
                tokens.push(Token::Plus);
                i += 1;
            }
            b'-' => {
                tokens.push(Token::Minus);
                i += 1;
            }
            b'*' => {
                tokens.push(Token::Star);
                i += 1;
            }
            b'%' => {
                tokens.push(Token::Percent);
                i += 1;
            }
            b'(' => {
                tokens.push(Token::LParen);
                i += 1;
            }
            b')' => {
                tokens.push(Token::RParen);
                i += 1;
            }
            b',' => {
                tokens.push(Token::Comma);
                i += 1;
            }
            b'/' => {
                if bytes.get(i + 1) == Some(&b'/') {
                    tokens.push(Token::SlashSlash);
                    i += 2;
                } else {
                    return Err(syntax(
                        "`/` is not an operator; use `//` (floor division)".into(),
                    ));
                }
            }
            b'0'..=b'9' => {
                let start = i;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                let digits = &text[start..i];
                let v: i64 = digits.parse().map_err(|_| {
                    syntax(format!("integer `{}` doesn't fit in i64", truncate(digits)))
                })?;
                tokens.push(Token::Int(v));
            }
            b if b.is_ascii_alphabetic() || b == b'_' => {
                let start = i;
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
                tokens.push(Token::Ident(text[start..i].to_owned()));
            }
            _ => {
                let c = text[i..].chars().next().unwrap_or('?');
                return Err(syntax(format!(
                    "unexpected character `{}`",
                    c.escape_default()
                )));
            }
        }
    }
    Ok(tokens)
}

struct Parser<'a> {
    text: &'a str,
    tokens: Vec<Token>,
    pos: usize,
    depth: usize,
    is_declared: &'a dyn Fn(&str) -> bool,
}

impl Parser<'_> {
    fn syntax(&self, reason: &str) -> DimError {
        DimError::Syntax {
            expr: truncate(self.text),
            reason: reason.to_owned(),
        }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn eat(&mut self, t: &Token) -> bool {
        if self.peek() == Some(t) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn enter(&mut self) -> Result<(), DimError> {
        self.depth += 1;
        if self.depth > limits::MAX_DIM_EXPR_DEPTH {
            return Err(DimError::TooDeep {
                max: limits::MAX_DIM_EXPR_DEPTH,
            });
        }
        Ok(())
    }

    fn expr(&mut self) -> Result<DimExpr, DimError> {
        let mut acc = self.term()?;
        loop {
            if self.eat(&Token::Plus) {
                let rhs = self.term()?;
                acc = acc.add(&rhs)?;
            } else if self.eat(&Token::Minus) {
                let rhs = self.term()?;
                acc = acc.sub(&rhs)?;
            } else {
                return Ok(acc);
            }
        }
    }

    fn term(&mut self) -> Result<DimExpr, DimError> {
        let mut acc = self.factor()?;
        loop {
            if self.eat(&Token::Star) {
                let rhs = self.factor()?;
                acc = acc.mul(&rhs)?;
            } else if self.eat(&Token::SlashSlash) {
                let k = self.divisor("//")?;
                acc = acc.floor_div(k)?;
            } else if self.eat(&Token::Percent) {
                let k = self.divisor("%")?;
                acc = acc.rem(k)?;
            } else {
                return Ok(acc);
            }
        }
    }

    fn divisor(&mut self, op: &'static str) -> Result<i64, DimError> {
        let d = self.factor()?;
        match d.as_literal() {
            Some(k) if k >= 1 => Ok(k),
            _ => Err(DimError::InvalidDivisor {
                op,
                divisor: d.to_string(),
            }),
        }
    }

    fn factor(&mut self) -> Result<DimExpr, DimError> {
        let Some(tok) = self.peek().cloned() else {
            return Err(self.syntax("unexpected end of the expression"));
        };
        self.pos += 1;
        match tok {
            Token::Int(v) => DimExpr::literal(v),
            Token::LParen => {
                self.enter()?;
                let e = self.expr()?;
                if !self.eat(&Token::RParen) {
                    return Err(self.syntax("missing `)`"));
                }
                self.depth -= 1;
                Ok(e)
            }
            Token::Ident(name) if name == "min" || name == "max" => {
                if !self.eat(&Token::LParen) {
                    return Err(self.syntax("`min` and `max` are functions and need `(`"));
                }
                self.enter()?;
                let a = self.expr()?;
                if !self.eat(&Token::Comma) {
                    return Err(self.syntax("`min`/`max` take two arguments separated by `,`"));
                }
                let b = self.expr()?;
                if !self.eat(&Token::RParen) {
                    return Err(self.syntax("missing `)`"));
                }
                self.depth -= 1;
                if name == "min" { a.min(&b) } else { a.max(&b) }
            }
            Token::Ident(name) => {
                if !names::is_symbol(&name) {
                    return Err(self.syntax(&format!(
                        "symbol `{}` is longer than {} bytes",
                        truncate(&name),
                        limits::MAX_SYMBOL_BYTES
                    )));
                }
                if !(self.is_declared)(&name) {
                    return Err(DimError::UndeclaredSymbol { name });
                }
                Ok(DimExpr::symbol(&name))
            }
            Token::Minus => Err(self.syntax("there is no unary minus")),
            _ => Err(self.syntax("expected an integer, a symbol, `(`, `min` or `max`")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn any(_: &str) -> bool {
        true
    }

    fn p(s: &str) -> DimExpr {
        DimExpr::parse(s, &any).unwrap_or_else(|e| panic!("{s}: {e}"))
    }

    fn canon(s: &str) -> String {
        p(s).to_string()
    }

    #[test]
    fn canonical_strings() {
        assert_eq!(canon("past + seq"), "past + seq");
        assert_eq!(canon("seq + past"), "past + seq");
        assert_eq!(canon("seq*2 + past - 1"), "past + 2*seq - 1");
        assert_eq!(canon("(H + 1) // 2"), "(H + 1) // 2");
        assert_eq!(canon("min(seq, 10)"), "min(10, seq)");
        assert_eq!(canon("min(10, seq)"), "min(10, seq)");
        assert_eq!(canon("(2*seq + 1) // 2"), "seq");
        assert_eq!(canon("seq - seq"), "0");
        assert_eq!(canon("1 - seq"), "0 - seq + 1");
        assert_eq!(canon("b * s * b"), "b*b*s");
        assert_eq!(canon("b*s + s + 3 + b*b*s"), "b*b*s + b*s + s + 3");
        assert_eq!(canon("2 * ((H + 1) // 2)"), "2*((H + 1) // 2)");
        assert_eq!(canon("H // 2 * W"), "(H // 2)*W");
        assert_eq!(canon("(3*H + 5) // 2"), "(H + 1) // 2 + H + 2");
        assert_eq!(canon("min(a, a + 1)"), "a");
        assert_eq!(canon("max(a, a + 1)"), "a + 1");
        assert_eq!(canon("7 // 2"), "3");
        assert_eq!(canon("(0 - 7) // 2"), "0 - 4");
        assert_eq!(canon("(0 - 7) % 2"), "1");
        assert_eq!(canon("seq % 4"), "seq - 4*(seq // 4)");
        assert_eq!(canon("seq // 1"), "seq");
        assert_eq!(canon(" ( a ) "), "a");
    }

    #[test]
    fn canonical_strings_parse_back() {
        for s in [
            "past + seq",
            "(H + 2*3 - 2*(3 - 1) - 1) // 2 + 1",
            "min(seq, 10) * 2 - max(a, b)",
            "(a*b + 3) // 4 * c",
            "1 - seq",
            "seq % 4 + (seq // 3) // 5",
            "max(min(a, b), c // 2)",
        ] {
            let e = p(s);
            let back = p(&e.to_string());
            assert_eq!(e, back, "{s} -> {e}");
        }
    }

    #[test]
    fn equality_is_normal_form_equality() {
        assert_eq!(p("batch * seq"), p("seq*batch"));
        assert_eq!(p("(a + b) * (a + b)"), p("a*a + 2*a*b + b*b"));
        assert_ne!(p("(seq // 2) * 2"), p("seq"));
        assert_eq!(p("(H + 2*1 - 1*(3 - 1) - 1) // 1 + 1"), p("H"));
    }

    #[test]
    fn literals_and_symbols() {
        assert_eq!(p("3 * 4").as_literal(), Some(12));
        assert_eq!(p("seq").as_symbol(), Some("seq"));
        assert_eq!(p("2*seq").as_symbol(), None);
        let syms: Vec<_> = p("min(a, b // 2) + c").symbols().into_iter().collect();
        assert_eq!(syms, ["a", "b", "c"]);
    }

    #[test]
    fn eval() {
        let e = p("(H + 2*1 - 1*(3 - 1) - 1) // 2 + 1");
        let v = e.eval(&|s| (s == "H").then_some(28)).unwrap();
        assert_eq!(v, 14);
        assert_eq!(p("seq % 4").eval(&|_| Some(10)).unwrap(), 2);
        assert_eq!(
            p("min(a, b)")
                .eval(&|s| Some(if s == "a" { 3 } else { 2 }))
                .unwrap(),
            2
        );
        assert!(matches!(p("a").eval(&|_| None), Err(EvalError::Unbound(_))));
        assert_eq!(
            p("a * a").eval(&|_| Some(i64::MAX)),
            Err(EvalError::Overflow)
        );
    }

    #[test]
    fn errors() {
        let err = |s: &str| DimExpr::parse(s, &|n| n != "undeclared").unwrap_err();
        assert!(matches!(err(""), DimError::Syntax { .. }));
        assert!(matches!(err("a +"), DimError::Syntax { .. }));
        assert!(matches!(err("-a"), DimError::Syntax { .. }));
        assert!(matches!(err("a / 2"), DimError::Syntax { .. }));
        assert!(matches!(err("(a"), DimError::Syntax { .. }));
        assert!(matches!(err("a)"), DimError::Syntax { .. }));
        assert!(matches!(err("min(a)"), DimError::Syntax { .. }));
        assert!(matches!(err("min"), DimError::Syntax { .. }));
        assert!(matches!(err("a $ b"), DimError::Syntax { .. }));
        assert!(matches!(err("ä"), DimError::Syntax { .. }));
        assert!(matches!(
            err("99999999999999999999"),
            DimError::Syntax { .. }
        ));
        assert!(matches!(
            err("undeclared + 1"),
            DimError::UndeclaredSymbol { .. }
        ));
        assert!(matches!(err("a // b"), DimError::InvalidDivisor { .. }));
        assert!(matches!(err("a // 0"), DimError::InvalidDivisor { .. }));
        assert!(matches!(
            err("a % (1 - 1)"),
            DimError::InvalidDivisor { .. }
        ));
        assert!(matches!(err("9223372036854775807 + 1"), DimError::Overflow));
        assert!(matches!(
            err("4611686018427387904 * a * 2"),
            DimError::Overflow
        ));
        assert!(matches!(
            err("0 - 9223372036854775807 - 1"),
            DimError::Overflow
        ));
        assert_eq!(
            err("9223372036854775807 + 1").code(),
            ErrorCode::DimOverflow
        );
        assert_eq!(err("a // b").code(), ErrorCode::InvalidDimExpr);
    }

    #[test]
    fn limits() {
        // Length: exactly 256 bytes is fine, 257 isn't.
        let ok = format!("a{}", " ".repeat(255));
        assert!(DimExpr::parse(&ok, &any).is_ok());
        let long = format!("a{}", " ".repeat(256));
        assert!(matches!(
            DimExpr::parse(&long, &any),
            Err(DimError::TooLong { .. })
        ));
        // Depth: 16 nested parentheses are fine, 17 aren't; min/max calls count too.
        let nest = |n: usize| format!("{}a{}", "(".repeat(n), ")".repeat(n));
        assert!(DimExpr::parse(&nest(16), &any).is_ok());
        assert!(matches!(
            DimExpr::parse(&nest(17), &any),
            Err(DimError::TooDeep { .. })
        ));
        let calls = |n: usize| format!("{}a{}", "min(".repeat(n), ", b)".repeat(n));
        assert!(DimExpr::parse(&calls(16), &any).is_ok());
        assert!(matches!(
            DimExpr::parse(&calls(17), &any),
            Err(DimError::TooDeep { .. })
        ));
        // Degree: 8 is fine, 9 isn't.
        assert!(DimExpr::parse(&["a"; 8].join("*"), &any).is_ok());
        assert!(matches!(
            DimExpr::parse(&["a"; 9].join("*"), &any),
            Err(DimError::DegreeTooHigh { .. })
        ));
        // Terms: (a+b+c+d+e+f+g+h)^2 has 36 terms, ^3 has 120.
        let s8 = "(a + b + c + d + e + f + g + h)";
        assert_eq!(p(&format!("{s8}*{s8}")).term_count(), 36);
        assert!(matches!(
            DimExpr::parse(&format!("{s8}*{s8}*{s8}"), &any),
            Err(DimError::TooManyTerms { .. })
        ));
        // 64 distinct symbols in one sum is the maximum.
        let syms: Vec<String> = (0..64).map(|i| format!("s{i}")).collect();
        let s64 = syms.join("+");
        assert!(s64.len() <= 256);
        assert_eq!(p(&s64).term_count(), 64);
        let s65 = format!("{s64}+t");
        assert!(matches!(
            DimExpr::parse(&s65, &any),
            Err(DimError::TooManyTerms { .. })
        ));
        // A symbol longer than 64 bytes.
        assert!(matches!(
            DimExpr::parse(&"s".repeat(65), &any),
            Err(DimError::Syntax { .. })
        ));
    }
}
