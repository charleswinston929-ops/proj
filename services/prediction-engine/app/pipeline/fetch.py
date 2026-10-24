"""Refresh raw results from the provider. Usage: python -m app.pipeline.fetch [season ...]   (default: current season only)"""
import sys
from ..providers.openfootball import OpenFootballProvider
def main():
    p = OpenFootballProvider("../../data/raw"); seasons = sys.argv[1:] or [p.seasons()[-1]]
    for s in seasons: print(s, len(p.get_matches(s, refresh=True)), "fixtures refreshed")
if __name__ == "__main__": main()
