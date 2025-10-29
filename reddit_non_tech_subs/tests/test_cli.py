from reddit_non_tech_subs.collector import parse_args, subs_from_arg


def test_cli_parsing_defaults():
    args = parse_args([])
    assert args.mode in {"new", "top"}
    assert args.posts > 0
    assert args.comments > 0


def test_cli_subs_parsing_override():
    args = parse_args(["--subs", "foo,bar , baz ", "--dry-run"])
    subs = subs_from_arg(args.subs)
    assert subs == ["foo", "bar", "baz"]
    assert args.dry_run is True


