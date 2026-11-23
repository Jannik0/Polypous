"""Tests of polyrng.py against the known values of graph-ir.md §3.1.12 and the statistics of
testing-strategy.md §3.8 (with its "Python value" column). Standard library only."""

from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import polyrng  # noqa: E402

DROP_LABEL = "dropout:layers[3]/drop"
N_DRAWS = 1 << 20


class KnownValues(unittest.TestCase):
    """graph-ir.md §3.1.12, "Known values" (bit-exact)."""

    def test_core(self) -> None:
        self.assertEqual(polyrng.draw(0, 0), 0xE220A8397B1DCDAF)
        self.assertEqual(polyrng.fnv1a("a"), 0xAF63DC4C8601EC8C)
        key = polyrng.stream_key(42, 7, DROP_LABEL)
        self.assertEqual(key, 0x559FE2396A2BE18C)
        draws = [polyrng.draw(key, i) for i in range(4)]
        self.assertEqual(draws, [0xDEAFBD2049BBA567, 0x97B9C0C7F5A35D08, 0xC5DA3B226895E44F, 0x09407FF20490E439])
        self.assertEqual(
            [polyrng.u24(z) for z in draws],
            [0.8698690533638, 0.5926780700683594, 0.7728611826896667, 0.036140382289886475],
        )

    def test_init_stream(self) -> None:
        key = polyrng.stream_key(42, 0, "init:fc1.weight")
        self.assertEqual(key, 0x4A64D327A84495D0)
        self.assertEqual(
            [polyrng.u53(polyrng.draw(key, i)) for i in range(3)],
            [0.9418494217379093, 0.952777646315502, 0.21886262424309422],
        )
        normals = [polyrng.normal(key, i) for i in range(3)]
        for got, want in zip(normals, [0.33102468465391816, 1.1951877566467608, 0.2241557644014358]):
            # ln and cos are the platform's: within 1 ulp of f64.
            self.assertLessEqual(abs(got - want), math.ulp(want), (got, want))

    def test_kaiming_uniform_table_case(self) -> None:
        shape = [128, 784]
        fans = polyrng.fans_of("nn.linear", "weight", shape)
        self.assertEqual(fans, (784, 128))
        init = {"kind": "kaiming_uniform"}
        # Only elements 0-2 are needed; the values don't depend on the shape.
        values = polyrng.initial_values(42, "fc1.weight", [3], init, fans)
        want = [polyrng.to_f32(v) for v in (0.077307545, 0.07921959, -0.049188793)]
        self.assertEqual(values, want)

    def test_dropout_mask(self) -> None:
        self.assertEqual(polyrng.dropout_mask(42, 7, "layers[3]/drop", [2, 2], 0.5), [True, True, True, False])
        self.assertEqual(polyrng.dropout_mask(42, 7, "x", [3], 0.0), [True, True, True])
        with self.assertRaises(ValueError):
            polyrng.dropout_mask(42, 7, "x", [3], 1.0)


class Rounding(unittest.TestCase):
    def test_to_f32(self) -> None:
        self.assertEqual(polyrng.to_f32(0.1), 0.10000000149011612)
        self.assertEqual(polyrng.to_f32(3.4028235677973366e38), math.inf)  # rounds up past f32 max
        self.assertEqual(polyrng.to_f32(-1e39), -math.inf)
        self.assertEqual(polyrng.to_f32(3.4028234663852886e38), 3.4028234663852886e38)
        # Ties to even: 1 + 2^-24 lies halfway between 1 and 1 + 2^-23.
        self.assertEqual(polyrng.to_f32(1.0 + 2.0**-24), 1.0)

    def test_fans(self) -> None:
        self.assertEqual(polyrng.fans_of("nn.conv2d", "weight", [16, 4, 3, 5]), (60, 240))
        self.assertEqual(polyrng.fans_of("nn.embedding", "weight", [1000, 64]), (64, 1000))
        self.assertIsNone(polyrng.fans_of("nn.linear", "bias", [4]))


class Statistics(unittest.TestCase):
    """testing-strategy.md §3.8, "Generator, statistics": each within 5 SE, chi-square at most
    `d + 5·sqrt(2d)`; the values must equal the table's Python column (rounded as printed)."""

    @classmethod
    def setUpClass(cls) -> None:
        key = polyrng.stream_key(42, 7, DROP_LABEL)
        cls.u = [polyrng.u24(polyrng.draw(key, i)) for i in range(N_DRAWS)]
        key8 = polyrng.stream_key(42, 8, DROP_LABEL)
        cls.u8 = [polyrng.u24(polyrng.draw(key8, i)) for i in range(N_DRAWS)]

    def test_u24_uniformity_and_mean(self) -> None:
        counts = [0] * 256
        for v in self.u:
            counts[int(v * 256)] += 1
        expected = N_DRAWS / 256
        chi2 = sum((c - expected) ** 2 / expected for c in counts)
        self.assertLessEqual(chi2, 255 + 5 * math.sqrt(2 * 255))
        self.assertEqual(round(chi2, 1), 241.8)
        se = math.sqrt(1 / 12 / N_DRAWS)
        z = abs(sum(self.u) / N_DRAWS - 0.5) / se
        self.assertEqual(round(z, 2), 0.73)

    def test_keep_rates_and_independence(self) -> None:
        worst = 0.0
        for p in (0.1, 0.25, 0.5, 0.9):
            kept = sum(1 for v in self.u if v >= p) / N_DRAWS
            z = abs(kept - (1 - p)) / math.sqrt(p * (1 - p) / N_DRAWS)
            worst = max(worst, z)
        self.assertLessEqual(worst, 5)
        self.assertEqual(round(worst, 2), 0.58)
        both = sum(1 for a, b in zip(self.u, self.u8) if a >= 0.5 and b >= 0.5) / N_DRAWS
        z = abs(both - 0.25) / math.sqrt(0.25 * 0.75 / N_DRAWS)
        self.assertEqual(round(z, 2), 0.63)

    def test_normal_moments(self) -> None:
        n = 1 << 19
        key = polyrng.stream_key(42, 0, "init:fc1.weight")
        xs = [polyrng.normal(key, i) for i in range(n)]
        mean = sum(xs) / n
        m2 = sum((x - mean) ** 2 for x in xs) / n
        m3 = sum((x - mean) ** 3 for x in xs) / n
        m4 = sum((x - mean) ** 4 for x in xs) / n
        zs = [
            abs(mean) / math.sqrt(1 / n),
            abs(m2 - 1) / math.sqrt(2 / n),
            abs(m3 / m2**1.5) / math.sqrt(6 / n),
            abs(m4 / m2**2 - 3) / math.sqrt(24 / n),
        ]
        self.assertLessEqual(max(zs), 5)
        self.assertEqual(round(max(zs), 2), 1.33)


if __name__ == "__main__":
    unittest.main()
