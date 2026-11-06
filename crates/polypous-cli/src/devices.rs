//! `polyp devices`: the devices Polypous found, as a table or as JSON (public-api.md §1.2).

use polypous::{DeviceInfo, ProbeReport};

/// Human-readable size: `-` if unknown, else MiB below 1 GiB, else GiB with two decimals.
pub(crate) fn format_memory(bytes: Option<u64>) -> String {
    const MIB: u64 = 1 << 20;
    const GIB: u64 = 1 << 30;
    match bytes {
        None => "-".to_string(),
        Some(bytes) if bytes < GIB => format!("{} MiB", bytes / MIB),
        // Display only: f64 is exact enough for two decimals.
        Some(bytes) => format!("{:.2} GiB", bytes as f64 / GIB as f64),
    }
}

/// The table `polyp devices` prints: one row per device, the primary GPU marked with `*`, the
/// physical GPU number in the `GPU` column (rows with the same number are one card), and a line
/// naming the primary GPU (or saying that only the CPU is used).
pub(crate) fn render_table(devices: &[DeviceInfo]) -> String {
    if devices.is_empty() {
        return "No devices found.\n".to_string();
    }
    let header = [
        "",
        "GPU",
        "BACKEND",
        "NAME",
        "KIND",
        "DEDICATED",
        "UNIFIED",
        "STATUS",
    ];
    let rows: Vec<[String; 8]> = devices
        .iter()
        .map(|device| {
            [
                if device.primary { "*" } else { "" }.to_string(),
                // Rows with the same number are one physical GPU (public-api.md §1.2).
                device
                    .physical_gpu
                    .map_or_else(|| "-".to_string(), |gpu| gpu.to_string()),
                device.backend.to_string(),
                device.name.clone(),
                device.kind.to_string(),
                format_memory(device.dedicated_memory),
                if device.unified_memory { "yes" } else { "no" }.to_string(),
                // Available, or unavailable with the reason (placement.md §1 wording).
                device.status.to_string(),
            ]
        })
        .collect();

    let mut widths = header.map(|title| title.chars().count());
    // The marker column is always one character wide, so the layout doesn't shift without a GPU.
    widths[0] = 1;
    for row in &rows {
        for (width, cell) in widths.iter_mut().zip(row) {
            *width = (*width).max(cell.chars().count());
        }
    }

    let mut out = String::new();
    let mut push_row = |cells: [&str; 8]| {
        let mut line = String::new();
        for (column, cell) in cells.iter().enumerate() {
            if column > 0 {
                line.push_str("  ");
            }
            if column + 1 == cells.len() {
                line.push_str(cell);
            } else {
                line.push_str(&format!("{cell:<width$}", width = widths[column]));
            }
        }
        out.push_str(line.trim_end());
        out.push('\n');
    };
    push_row(header);
    for row in &rows {
        push_row(row.each_ref().map(String::as_str));
    }

    out.push('\n');
    match devices.iter().find(|device| device.primary) {
        Some(primary) => out.push_str(&format!(
            "* primary GPU: {} ({})\n",
            primary.name, primary.id
        )),
        None => out.push_str("No usable GPU: Polypous runs on the CPU only.\n"),
    }
    out
}

/// How the devices were probed, below the table: isolated or not, prober processes started,
/// cache hits, and the report's warnings.
pub(crate) fn render_probe_summary(report: &ProbeReport) -> String {
    let isolation = if report.isolated {
        "isolated (separate prober process)"
    } else {
        "NOT isolated (in-process probing with crash journal)"
    };
    let mut out = format!(
        "Probing: {isolation}; prober processes started: {}; from the probe cache: {} of {} devices\n",
        report.prober_starts,
        report.cache_hits.len(),
        report.devices.len()
    );
    for warning in &report.warnings {
        out.push_str(&format!("warning: {warning}\n"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic devices of the reference machine, read from the facade's JSON (the public types
    /// can only be built from JSON outside the facade, by design).
    fn reference_machine() -> Vec<DeviceInfo> {
        let device = |id: &str,
                      backend: &str,
                      name: &str,
                      kind: &str,
                      memory: &str,
                      unified: bool,
                      status: &str,
                      physical_gpu: &str,
                      primary: bool| {
            format!(
                r#"{{"id":{id},"backend":"{backend}","name":"{name}","vendor":null,"driver":null,
                "kind":"{kind}","dedicated_memory":{memory},"free_memory":null,
                "unified_memory":{unified},
                "identity":{{"pci_bus_id":null,"luid":null,"registry_id":null}},
                "capabilities":{{"f16":false,"bf16":false,"subgroup_size_min":null,
                "subgroup_size_max":null,"max_buffer_size":null,"max_shared_memory":null,
                "compute_units":null}},"status":{status},"physical_gpu":{physical_gpu},
                "primary":{primary}}}"#
            )
        };
        let available = r#"{"state":"available","isolated":false}"#;
        let json = format!(
            "[{},{},{},{}]",
            device(
                r#"{"backend":"cuda","index":0}"#,
                "cuda",
                "NVIDIA GeForce RTX 3060 Laptop GPU",
                "discrete",
                "6442450944",
                false,
                available,
                "0",
                true
            ),
            device(
                r#"{"backend":"vulkan","adapter":"discrete_gpu","index":0}"#,
                "vulkan",
                "NVIDIA GeForce RTX 3060 Laptop GPU",
                "discrete",
                "6442450944",
                false,
                available,
                "0",
                false
            ),
            device(
                r#"{"backend":"vulkan","adapter":"integrated_gpu","index":0}"#,
                "vulkan",
                "Intel(R) Iris(R) Xe Graphics",
                "integrated",
                "536870912",
                true,
                r#"{"state":"unavailable","reason":{"code":"crashed_during_probe"},"isolated":true}"#,
                "1",
                false
            ),
            device(
                r#"{"backend":"cpu"}"#,
                "cpu",
                "12th Gen Intel(R) Core(TM) i7-12800H",
                "cpu",
                "null",
                true,
                available,
                "null",
                false
            ),
        );
        serde_json::from_str(&json).unwrap()
    }

    #[test]
    fn table_snapshot() {
        let table = render_table(&reference_machine());
        // (The first line starts with spaces, so it is not continued with `\`.)
        let expected = "   GPU  BACKEND  NAME                                  KIND        DEDICATED  UNIFIED  STATUS
*  0    cuda     NVIDIA GeForce RTX 3060 Laptop GPU    discrete    6.00 GiB   no       available, not isolated
   0    vulkan   NVIDIA GeForce RTX 3060 Laptop GPU    discrete    6.00 GiB   no       available, not isolated
   1    vulkan   Intel(R) Iris(R) Xe Graphics          integrated  512 MiB    yes      unavailable: crashed during probe
   -    cpu      12th Gen Intel(R) Core(TM) i7-12800H  cpu         -          yes      available, not isolated

* primary GPU: NVIDIA GeForce RTX 3060 Laptop GPU (cuda:0)
";
        assert_eq!(table, expected);
    }

    #[test]
    fn cpu_only_table() {
        let devices: Vec<DeviceInfo> = reference_machine().into_iter().skip(3).collect();
        let table = render_table(&devices);
        let expected =
            "   GPU  BACKEND  NAME                                  KIND  DEDICATED  UNIFIED  STATUS
   -    cpu      12th Gen Intel(R) Core(TM) i7-12800H  cpu   -          yes      available, not isolated

No usable GPU: Polypous runs on the CPU only.
";
        assert_eq!(table, expected);
    }

    #[test]
    fn probe_summary() {
        let report: ProbeReport = serde_json::from_str(
            r#"{"devices":[],"isolated":false,"cache_hits":[],"prober_starts":0,
                "fingerprint_known":true,"warnings":["vulkan:cpu:0 crashed this app"]}"#,
        )
        .unwrap();
        assert_eq!(
            render_probe_summary(&report),
            "Probing: NOT isolated (in-process probing with crash journal); prober processes \
             started: 0; from the probe cache: 0 of 0 devices\nwarning: vulkan:cpu:0 crashed this app\n"
        );
    }

    #[test]
    fn empty_table() {
        assert_eq!(render_table(&[]), "No devices found.\n");
    }

    #[test]
    fn memory_formatting() {
        assert_eq!(format_memory(None), "-");
        assert_eq!(format_memory(Some(0)), "0 MiB");
        assert_eq!(format_memory(Some(512 << 20)), "512 MiB");
        assert_eq!(format_memory(Some(1 << 30)), "1.00 GiB");
        assert_eq!(format_memory(Some(6_441_926_656)), "6.00 GiB");
        assert_eq!(format_memory(Some(u64::MAX)), "17179869184.00 GiB");
    }

    #[test]
    fn wide_names_align_by_characters() {
        let mut devices = reference_machine();
        devices.truncate(1);
        devices[0].name = "Größe™ GPU".to_string();
        let table = render_table(&devices);
        let lines: Vec<&str> = table.lines().collect();
        let column = |line: &str, text: &str| line[..line.find(text).unwrap()].chars().count();
        // "KIND" in the header and "discrete" in the row start in the same column.
        assert_eq!(column(lines[0], "KIND"), column(lines[1], "discrete"));
    }
}
