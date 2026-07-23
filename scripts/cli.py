#!/usr/bin/env python3
"""Football Predictor — 此地无垠足球预测 CLI (Python)"""
import argparse, json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def cmd_predict(args):
    """Predict match outcome."""
    try:
        from scripts.football_predictor import predict_match
        result = predict_match(args.home, args.away)
        print(json.dumps({"home": args.home, "away": args.away, "prediction": str(result)[:300], "status": "ok"}, ensure_ascii=False, indent=2))
    except ImportError:
        # Fallback to node.js predictor
        import subprocess
        node_script = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'predict.mjs')
        if os.path.exists(node_script):
            result = subprocess.run(['node', node_script, args.home, args.away], capture_output=True, text=True, timeout=30)
            print(result.stdout)
            if result.returncode != 0:
                print(result.stderr, file=sys.stderr)
        else:
            print(json.dumps({"error": "no predictor available"}, ensure_ascii=False))

def cmd_leagues(args):
    """List supported leagues."""
    try:
        from scripts.football_predictor import list_leagues
        list_leagues()  # prints directly, returns None
    except (ImportError, AttributeError):
        # Fallback
        import subprocess
        node_script = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'predict.mjs')
        if os.path.exists(node_script):
            result = subprocess.run(['node', node_script, '--list'], capture_output=True, text=True, timeout=10)
            print(result.stdout or "Run: node scripts/predict.mjs <home> <away>")
        else:
            print("  premier-league, la-liga, serie-a, bundesliga, ligue-1")

def cmd_info(args):
    """Show product info."""
    print(json.dumps({
        "product": "此地无垠 Football Predictor",
        "engines": ["Elo+DC", "Monte Carlo", "Bayesian XG"],
        "languages": ["Python", "Node.js"],
        "status": "ok"
    }, ensure_ascii=False, indent=2))

def main():
    p = argparse.ArgumentParser(description='Football Predictor 足球预测工具')
    sub = p.add_subparsers(dest='command')

    pr = sub.add_parser('predict', help='预测比赛结果')
    pr.add_argument('home', help='主队')
    pr.add_argument('away', help='客队')
    pr.add_argument('--neutral', action='store_true', help='中立场')

    sub.add_parser('leagues', help='列出支持的联赛')
    sub.add_parser('info', help='产品信息')

    args = p.parse_args()
    if args.command == 'predict': cmd_predict(args)
    elif args.command == 'leagues': cmd_leagues(args)
    elif args.command == 'info': cmd_info(args)
    else: p.print_help()

if __name__ == '__main__':
    main()
