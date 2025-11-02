import logging
import time
from typing import List, Optional
from reddit_non_tech_subs.collector import LOGGER, FetchConfig, fetch, build_dataframe, add_sentiment_columns, parse_args, subs_from_arg
from reddit_non_tech_subs.io_utils import append_run_log, ensure_dir, load_env, write_outputs


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

def main(argv: Optional[List[str]] = None) -> int:
    _setup_logging()
    load_env()
    args = parse_args(argv)
    subs = subs_from_arg(args.subs)
    cfg = FetchConfig(
        subreddits=subs,
        mode=args.mode,
        time_filter=args.time_filter,
        posts_per_sub=args.posts,
        top_level_comment_limit=args.comments,
        use_hf=bool(args.hf),
        outdir=args.outdir,
        search_ai=not bool(args.no_ai_filter),
        dry_run=bool(args.dry_run),
    )
    ensure_dir(cfg.outdir)
    LOGGER.info(
        "Starting collection: subs=%d mode=%s tf=%s posts=%d comments=%d hf=%s",
        len(cfg.subreddits), cfg.mode, cfg.time_filter, cfg.posts_per_sub, cfg.top_level_comment_limit, cfg.use_hf,
    )

    if cfg.dry_run:
        LOGGER.info("Dry run complete: %s", cfg)
        return 0

    t0 = time.time()
    posts, comments = fetch(cfg)
    df = build_dataframe(posts, comments)
    df = add_sentiment_columns(df, cfg.use_hf)

    base_name = "reddit_ai_nontech"
    outputs = write_outputs(df, cfg.outdir, base_name)
    elapsed = time.time() - t0
    LOGGER.info("Wrote CSV to %s and Parquet to %s (%d rows) in %.1fs", outputs.get("csv"), outputs.get("parquet"), len(df), elapsed)

    append_run_log(
        cfg.outdir,
        {
            "subs_count": len(cfg.subreddits),
            "rows": len(df),
            "mode": cfg.mode,
            "time_filter": cfg.time_filter,
            "use_hf": cfg.use_hf,
            "elapsed_sec": round(elapsed, 2),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# python -m reddit_tech_subs.main --dry-run
# python -m reddit_tech_subs.main --posts 1000 --comments 100 --outdir reddit_tech_subs/data


# python -m reddit_non_tech_subs.collector --posts 1000 --comments 100 --outdir reddit_non_tech_subs/data