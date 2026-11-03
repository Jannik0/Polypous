#!/bin/bash
# Setup script for Claude Code cloud environments (Ubuntu 24.04, runs as root, no GPU).
# Paste into the environment's "Setup script" field, or load this file there.
# The result is cached when it finishes in about five minutes, so keep it lean.
# See docs/dev-environment.md#cloud-environments-for-agents.
set -euo pipefail

# Software Vulkan driver (Mesa lavapipe), so Vulkan code paths can be tested without a GPU.
apt-get update
apt-get install -y --no-install-recommends mesa-vulkan-drivers vulkan-tools libvulkan1

# Lint tools for the default toolchain. The pinned toolchain from rust-toolchain.toml
# (added in M0) is installed by rustup on the first cargo invocation.
rustup component add rustfmt clippy

# cargo-deny (ADR-0010) is deliberately not installed here: building it from source would
# push the script past the caching limit. CI runs it on every PR.

# Commits are authored with the owner's GitHub no-reply address, not an agent identity
# (CLAUDE.md, "Conventions"). Otherwise GitHub adds the agent as co-author when squash-merging.
git config --global user.name "Jannik0"
git config --global user.email "38440534+Jannik0@users.noreply.github.com"
