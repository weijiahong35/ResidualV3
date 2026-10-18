#!/usr/bin/env python3
"""Print overall generation progress or verify all outputs against prompt IDs."""
import argparse
import json
from pathlib import Path
from dispatch import DEFAULT_POOL, pool_progress, verify_outputs
from profiles import PROFILES, add_profile_argument


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool-dir', type=Path, default=DEFAULT_POOL)
    parser.add_argument('--generation-dir', type=Path)
    add_profile_argument(parser)
    parser.add_argument('--verify', action='store_true', help='Full pinned-ID and profile length validation; run after workers finish')
    parser.add_argument('--details', action='store_true', help='Include every shard')
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    generation_dir = args.generation_dir or args.pool_dir / f'regenerated_ctx{profile.max_model_len}'
    kwargs = {'thinking': profile.mode}
    if args.verify:
        kwargs['max_model_len'] = profile.max_model_len
    result = (verify_outputs if args.verify else pool_progress)(args.pool_dir, generation_dir, **kwargs)
    if not args.details:
        result.pop('shards', None)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
