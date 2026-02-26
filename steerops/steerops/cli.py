"""
CLI for steerops.

Commands:
    steerops apply  <patch> --model <name> --prompt <text>
    steerops info   <patch>
    steerops validate <patch> --layers <num>
    steerops merge  <patch1> <patch2> -o <output>
"""

from __future__ import annotations

import argparse
import json
import sys

from steerops.patch import Patch


def main():
    parser = argparse.ArgumentParser(
        prog="steerops",
        description="SteerOps Activation Steering CLI",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ── apply ─────────────────────────────────────────────────
    apply_p = sub.add_parser("apply", help="Apply a patch and generate text")
    apply_p.add_argument("patch", help="Path to patch JSON file")
    apply_p.add_argument("--model", required=True, help="HF model name")
    apply_p.add_argument("--prompt", required=True, help="Input prompt")
    apply_p.add_argument("--max-tokens", type=int, default=200)
    apply_p.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu", "mps"]
    )
    apply_p.add_argument("--quantize", action="store_true")

    # ── info ──────────────────────────────────────────────────
    info_p = sub.add_parser("info", help="Show patch details")
    info_p.add_argument("patch", help="Path to patch JSON file")

    # ── validate ──────────────────────────────────────────────
    val_p = sub.add_parser("validate", help="Validate a patch")
    val_p.add_argument("patch", help="Path to patch JSON file")
    val_p.add_argument(
        "--layers", type=int, required=True,
        help="Number of layers in the target model",
    )

    # ── merge ─────────────────────────────────────────────────
    merge_p = sub.add_parser("merge", help="Merge multiple patches")
    merge_p.add_argument("patches", nargs="+", help="Patch files to merge")
    merge_p.add_argument("-o", "--output", required=True, help="Output file")
    merge_p.add_argument(
        "--name", default="merged", help="Name for merged patch"
    )
    merge_p.add_argument(
        "--strategy", default="average",
        choices=["average", "sum", "first"],
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "apply":
        cmd_apply(args)
    elif args.command == "info":
        cmd_info(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "merge":
        cmd_merge(args)


def cmd_apply(args):
    from steerops.steerer import Steerer

    print(f"📂 Loading patch: {args.patch}")
    print(f"🤖 Loading model: {args.model}")
    print(f"💬 Prompt: {args.prompt}")
    print()

    text = Steerer.run(
        patch_path=args.patch,
        model_name=args.model,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        device=args.device,
        quantize=args.quantize,
    )

    print("─" * 60)
    print("📝 Generated output:")
    print("─" * 60)
    print(text)


def cmd_info(args):
    patch = Patch.from_file(args.patch)
    print(patch.summary())
    print()
    print(f"Description: {patch.description or '(none)'}")

    # Show direction vector stats
    for iv in patch.interventions:
        vec_info = "None"
        if iv.direction_vector:
            vec_info = (
                f"dim={len(iv.direction_vector)}, "
                f"norm={sum(v**2 for v in iv.direction_vector)**0.5:.4f}"
            )
        print(f"  Layer {iv.layer}: strength={iv.strength:+.2f}, vector={vec_info}")


def cmd_validate(args):
    patch = Patch.from_file(args.patch)
    warnings = patch.validate_for_model(args.layers)

    if not warnings:
        print(f"✅ Patch '{patch.name}' is valid for {args.layers}-layer model")
    else:
        print(f"⚠️  Patch '{patch.name}' has {len(warnings)} warning(s):")
        for w in warnings:
            print(f"  - {w}")
        sys.exit(1)


def cmd_merge(args):
    patches = [Patch.from_file(p) for p in args.patches]
    merged = Patch.merge(patches, name=args.name, strategy=args.strategy)
    merged.save(args.output)

    print(f"✅ Merged {len(patches)} patches → {args.output}")
    print(f"   Strategy: {args.strategy}")
    print(f"   Interventions: {len(merged.interventions)}")


if __name__ == "__main__":
    main()
