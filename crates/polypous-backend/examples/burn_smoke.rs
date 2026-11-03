//! Smoke test for the pinned Burn version (issue #2, ADR-0007 follow-up notes).
//!
//! Not part of the library: device discovery and probing arrive with issue #4. This program only
//! answers the spike's questions on a given machine:
//!
//! 1. Does a binary built with a backend feature start when that backend's runtime libraries are
//!    missing? (It must, ADR-0003.)
//! 2. Does a tiny computation on that backend give the right result?
//!
//! ```text
//! cargo run -p polypous-backend --example burn_smoke --features cuda -- cuda
//! cargo run -p polypous-backend --example burn_smoke --features vulkan -- vulkan
//! cargo run -p polypous-backend --example burn_smoke --features vulkan -- vulkan 1   # second device
//! cargo run -p polypous-backend --example burn_smoke -- cpu
//! ```
//!
//! Exit code 0: the backend worked, or it has no devices on this machine (reported, not a
//! failure). Exit code 1: wrong result, or Burn panicked while using a device it had enumerated.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::process::ExitCode;

use burn::tensor::{Device, DeviceType, Tensor};

/// `[[1, 2], [3, 4]] @ [[1, 2], [3, 4]]`.
const EXPECTED: [f32; 4] = [7.0, 10.0, 15.0, 22.0];

fn main() -> ExitCode {
    let backend = std::env::args().nth(1).unwrap_or_else(|| "cpu".to_string());
    println!(
        "polypous burn_smoke: backend={backend}, os={}, arch={}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );

    let Some(device_type) = device_type(&backend) else {
        println!(
            "backend `{backend}` is not compiled in or unknown; enable its cargo feature \
             (cpu, cuda, vulkan, metal)"
        );
        return ExitCode::FAILURE;
    };

    // Step 1: enumeration. This must not panic when the runtime libraries are missing.
    let devices = match catch_unwind(|| {
        Device::enumerate(device_type)
            .into_iter()
            .collect::<Vec<_>>()
    }) {
        Ok(devices) => devices,
        Err(_) => {
            println!("FAIL: Burn panicked while enumerating `{backend}` devices");
            return ExitCode::FAILURE;
        }
    };
    println!("enumerated {} device(s)", devices.len());
    for (index, device) in devices.iter().enumerate() {
        println!("  [{index}] {device:?}");
    }
    // Optional second argument: index of the device to use (default 0), for machines with several
    // GPUs on one backend, such as an iGPU and a dGPU over Vulkan.
    let index = std::env::args()
        .nth(2)
        .and_then(|arg| arg.parse::<usize>().ok())
        .unwrap_or(0);
    let Some(device) = devices.into_iter().nth(index) else {
        println!("OK: no `{backend}` device on this machine; the binary started and kept running");
        return ExitCode::SUCCESS;
    };
    println!("using {device:?}");

    // Step 2: a tiny computation, read back to the host and compared with the expected values.
    let result = catch_unwind(AssertUnwindSafe(|| {
        let a = Tensor::<2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
        a.clone().matmul(a).into_data().try_to_vec::<f32>()
    }));
    match result {
        Ok(Ok(values)) if values == EXPECTED => {
            println!("OK: matmul result {values:?}");
            ExitCode::SUCCESS
        }
        Ok(Ok(values)) => {
            println!("FAIL: matmul result {values:?}, expected {EXPECTED:?}");
            ExitCode::FAILURE
        }
        Ok(Err(err)) => {
            println!("FAIL: could not read the result back: {err:?}");
            ExitCode::FAILURE
        }
        Err(_) => {
            println!(
                "FAIL: Burn panicked while running on {device:?} (see the panic message above)"
            );
            ExitCode::FAILURE
        }
    }
}

/// Maps a command-line name to Burn's device filter, if that backend is compiled in.
fn device_type(name: &str) -> Option<DeviceType> {
    match name {
        #[cfg(feature = "cpu")]
        "cpu" => Some(DeviceType::Flex),
        #[cfg(feature = "cuda")]
        "cuda" => Some(DeviceType::Cuda),
        #[cfg(feature = "vulkan")]
        "vulkan" => Some(DeviceType::Vulkan),
        #[cfg(feature = "metal")]
        "metal" => Some(DeviceType::Metal),
        _ => None,
    }
}
