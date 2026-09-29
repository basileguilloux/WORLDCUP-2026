"""Tournament formats for 10_tournament_sim.py: WC 2026 and the ten backtest tournaments.

One Format per tournament: the groups, how many advance, the knockout bracket as
group-position slots, the hosts and where each knockout match is played. Group
letters and brackets are transcribed from the official draws and regulations;
tests/test_tournament_formats.py checks them against results.csv (groups against
the group-stage fixtures, brackets against the real knockout pairings).

Slots: "1A" = winner of group A, "2C" = runner-up of group C, "3/B" = the
third-placed team that meets the winner of group B, as given by the
tournament's third-place allocation table.

Brackets are listed in BRACKET ORDER, not match-number order: the winners of
matches 0 and 1 meet in the next round, then 2 and 3, and so on. For the World
Cups this puts FIFA's round-of-16 matches 49, 50, 53, 54, 51, 52, 55, 56 in that
order, so quarter-finals 57-60 and semi-finals 61-62 come out as adjacent pairs.

UEFA's third-place allocation tables (15 combinations of qualifying thirds):
  * THIRDS_2016 -- Regulations of the UEFA European Football Championship
    2014-16, Article 17.03. Columns: the thirds facing 1A, 1B, 1C, 1D.
  * THIRDS_2020 -- Regulations 2018-20, Article 21.05, the same table used for
    Euro 2024. Columns: the thirds facing 1B, 1C, 1E, 1F.
"""
from dataclasses import dataclass, field

import pandas as pd

# combination of qualifying thirds -> group of the third facing each listed winner
THIRDS_2016_WINNERS = ("A", "B", "C", "D")
THIRDS_2016 = {
    "ABCD": "CDAB", "ABCE": "CABE", "ABCF": "CABF", "ABDE": "DABE", "ABDF": "DABF",
    "ABEF": "EABF", "ACDE": "CDAE", "ACDF": "CDAF", "ACEF": "CAFE", "ADEF": "DAFE",
    "BCDE": "CDBE", "BCDF": "CDBF", "BCEF": "ECBF", "BDEF": "EDBF", "CDEF": "CDFE",
}
THIRDS_2020_WINNERS = ("B", "C", "E", "F")
THIRDS_2020 = {
    "ABCD": "ADBC", "ABCE": "AEBC", "ABCF": "AFBC", "ABDE": "DEAB", "ABDF": "DFAB",
    "ABEF": "EFBA", "ACDE": "EDCA", "ACDF": "FDCA", "ACEF": "EFCA", "ADEF": "EFDA",
    "BCDE": "EDBC", "BCDF": "FDCB", "BCEF": "FECB", "BDEF": "FEDB", "CDEF": "FEDC",
}

WC_R16 = [("1A", "2B"), ("1C", "2D"), ("1E", "2F"), ("1G", "2H"),
          ("1B", "2A"), ("1D", "2C"), ("1F", "2E"), ("1H", "2G")]
EURO_R16_2016 = [("2A", "2C"), ("1D", "3/D"), ("1B", "3/B"), ("1F", "2E"),
                 ("1C", "3/C"), ("1E", "2D"), ("1A", "3/A"), ("2B", "2F")]
EURO_R16_2020 = [("1B", "3/B"), ("1A", "2C"), ("1F", "3/F"), ("2D", "2E"),
                 ("1E", "3/E"), ("1D", "2F"), ("1C", "3/C"), ("2A", "2B")]


@dataclass(frozen=True)
class Format:
    name: str
    tournament: str                     # `tournament` value in results.csv
    start: str                          # first match day
    end: str                            # last match day
    hosts: frozenset
    ko_rounds: tuple                    # rounds played, e.g. ("R16", "QF", "SF", "final")
    groups: dict = field(default_factory=dict)   # letter -> teams; empty = read off fixtures
    best_thirds: int = 0
    thirds_table: dict = None
    thirds_winners: tuple = ()
    bracket: list = None                # first knockout round, slot pairs in bracket order
    ko_venues: dict = None              # round -> venue country per match, bracket order
    seeded: bool = False                # WC 2026: approximate 1-32 seeding, see 10_tournament_sim
    cutoff: str = None                  # model sees data strictly before this date
    recent_since: str = None            # start of the FIFA-blend "recent matches" window
    ranking_snapshot: bool = False      # WC 2026 reads its own FIFA snapshot

    @property
    def cutoff_date(self):
        if self.cutoff:
            return pd.Timestamp(self.cutoff)
        return pd.Timestamp(self.start) - pd.Timedelta(days=1)

    @property
    def recent_start(self):
        if self.recent_since:
            return pd.Timestamp(self.recent_since)
        return self.cutoff_date - pd.DateOffset(years=4)

    @property
    def stages(self):
        """Ordered stage-reached categories, e.g. group, R16, QF, SF, runner-up, champion."""
        return ("group",) + tuple(self.ko_rounds[:-1]) + ("runner-up", "champion")

    @property
    def reach_columns(self):
        """Output columns after `qualify`: reaching each later round, then the title."""
        return tuple(self.ko_rounds[1:]) + ("champion",)

    def venue(self, rnd, i):
        """Country where match i (bracket order) of round `rnd` is played."""
        if self.ko_venues and rnd in self.ko_venues:
            return self.ko_venues[rnd][i]
        assert len(self.hosts) == 1, f"{self.name}: knockout venues needed for {rnd}"
        return next(iter(self.hosts))


def _wc(year, start, end, host, groups):
    return Format(name=f"World Cup {year}", tournament="FIFA World Cup", start=start, end=end,
                  hosts=frozenset({host}), ko_rounds=("R16", "QF", "SF", "final"),
                  groups=groups, bracket=WC_R16)


def _euro16(name, start, end, hosts, groups, bracket, table, winners, venues=None):
    return Format(name=name, tournament="UEFA Euro", start=start, end=end, hosts=frozenset(hosts),
                  ko_rounds=("R16", "QF", "SF", "final"), groups=groups, best_thirds=4,
                  thirds_table=table, thirds_winners=winners, bracket=bracket, ko_venues=venues)


def _g(**kw):
    return {k: tuple(v) for k, v in kw.items()}


WC2026 = Format(
    name="World Cup 2026", tournament="FIFA World Cup", start="2026-06-11", end="2026-07-19",
    hosts=frozenset({"United States", "Mexico", "Canada"}),
    ko_rounds=("R32", "R16", "QF", "SF", "final"), best_thirds=8, seeded=True,
    # the forecast used every scored row, through 11 June (matchday 1 included)
    cutoff="2026-06-12", recent_since="2022-06-01", ranking_snapshot=True)

BACKTEST = [
    _wc(2006, "2006-06-09", "2006-07-09", "Germany", _g(
        A=["Germany", "Costa Rica", "Poland", "Ecuador"],
        B=["England", "Paraguay", "Trinidad and Tobago", "Sweden"],
        C=["Argentina", "Ivory Coast", "Serbia", "Netherlands"],
        D=["Mexico", "Iran", "Angola", "Portugal"],
        E=["Italy", "Ghana", "United States", "Czech Republic"],
        F=["Brazil", "Croatia", "Australia", "Japan"],
        G=["France", "Switzerland", "South Korea", "Togo"],
        H=["Spain", "Ukraine", "Tunisia", "Saudi Arabia"])),
    _wc(2010, "2010-06-11", "2010-07-11", "South Africa", _g(
        A=["South Africa", "Mexico", "Uruguay", "France"],
        B=["Argentina", "Nigeria", "South Korea", "Greece"],
        C=["England", "United States", "Algeria", "Slovenia"],
        D=["Germany", "Australia", "Serbia", "Ghana"],
        E=["Netherlands", "Denmark", "Japan", "Cameroon"],
        F=["Italy", "Paraguay", "New Zealand", "Slovakia"],
        G=["Brazil", "North Korea", "Ivory Coast", "Portugal"],
        H=["Spain", "Switzerland", "Honduras", "Chile"])),
    _wc(2014, "2014-06-12", "2014-07-13", "Brazil", _g(
        A=["Brazil", "Croatia", "Mexico", "Cameroon"],
        B=["Spain", "Netherlands", "Chile", "Australia"],
        C=["Colombia", "Greece", "Ivory Coast", "Japan"],
        D=["Uruguay", "Costa Rica", "England", "Italy"],
        E=["Switzerland", "Ecuador", "France", "Honduras"],
        F=["Argentina", "Bosnia and Herzegovina", "Iran", "Nigeria"],
        G=["Germany", "Portugal", "Ghana", "United States"],
        H=["Belgium", "Algeria", "Russia", "South Korea"])),
    _wc(2018, "2018-06-14", "2018-07-15", "Russia", _g(
        A=["Russia", "Saudi Arabia", "Egypt", "Uruguay"],
        B=["Portugal", "Spain", "Morocco", "Iran"],
        C=["France", "Australia", "Peru", "Denmark"],
        D=["Argentina", "Iceland", "Croatia", "Nigeria"],
        E=["Brazil", "Switzerland", "Costa Rica", "Serbia"],
        F=["Germany", "Mexico", "Sweden", "South Korea"],
        G=["Belgium", "Panama", "Tunisia", "England"],
        H=["Poland", "Senegal", "Colombia", "Japan"])),
    _wc(2022, "2022-11-20", "2022-12-18", "Qatar", _g(
        A=["Qatar", "Ecuador", "Senegal", "Netherlands"],
        B=["England", "Iran", "United States", "Wales"],
        C=["Argentina", "Saudi Arabia", "Mexico", "Poland"],
        D=["France", "Australia", "Denmark", "Tunisia"],
        E=["Spain", "Costa Rica", "Germany", "Japan"],
        F=["Belgium", "Canada", "Morocco", "Croatia"],
        G=["Brazil", "Serbia", "Switzerland", "Cameroon"],
        H=["Portugal", "Ghana", "Uruguay", "South Korea"])),
    # Euro 2008: semi-finals are QF1 v QF2 and QF3 v QF4
    Format(name="Euro 2008", tournament="UEFA Euro", start="2008-06-07", end="2008-06-29",
           hosts=frozenset({"Austria", "Switzerland"}), ko_rounds=("QF", "SF", "final"),
           groups=_g(A=["Switzerland", "Czech Republic", "Portugal", "Turkey"],
                     B=["Austria", "Croatia", "Germany", "Poland"],
                     C=["Netherlands", "Italy", "Romania", "France"],
                     D=["Greece", "Sweden", "Spain", "Russia"]),
           bracket=[("1A", "2B"), ("1B", "2A"), ("1C", "2D"), ("1D", "2C")],
           ko_venues={"QF": ["Switzerland", "Austria", "Switzerland", "Austria"],
                      "SF": ["Switzerland", "Austria"], "final": ["Austria"]}),
    # Euro 2012: semi-finals are QF1 v QF3 and QF2 v QF4
    Format(name="Euro 2012", tournament="UEFA Euro", start="2012-06-08", end="2012-07-01",
           hosts=frozenset({"Poland", "Ukraine"}), ko_rounds=("QF", "SF", "final"),
           groups=_g(A=["Poland", "Greece", "Russia", "Czech Republic"],
                     B=["Netherlands", "Denmark", "Germany", "Portugal"],
                     C=["Spain", "Italy", "Republic of Ireland", "Croatia"],
                     D=["Ukraine", "Sweden", "France", "England"]),
           bracket=[("1A", "2B"), ("1C", "2D"), ("1B", "2A"), ("1D", "2C")],
           ko_venues={"QF": ["Poland", "Ukraine", "Poland", "Ukraine"],
                      "SF": ["Ukraine", "Poland"], "final": ["Ukraine"]}),
    _euro16("Euro 2016", "2016-06-10", "2016-07-10", {"France"}, _g(
        A=["France", "Romania", "Albania", "Switzerland"],
        B=["Wales", "Slovakia", "England", "Russia"],
        C=["Poland", "Northern Ireland", "Germany", "Ukraine"],
        D=["Spain", "Czech Republic", "Turkey", "Croatia"],
        E=["Belgium", "Italy", "Republic of Ireland", "Sweden"],
        F=["Portugal", "Iceland", "Austria", "Hungary"]),
        EURO_R16_2016, THIRDS_2016, THIRDS_2016_WINNERS),
    _euro16("Euro 2020", "2021-06-11", "2021-07-11",
            {"Azerbaijan", "Denmark", "England", "Germany", "Hungary", "Italy", "Netherlands",
             "Romania", "Russia", "Scotland", "Spain"}, _g(
        A=["Turkey", "Italy", "Wales", "Switzerland"],
        B=["Denmark", "Finland", "Belgium", "Russia"],
        C=["Netherlands", "Ukraine", "Austria", "North Macedonia"],
        D=["England", "Croatia", "Scotland", "Czech Republic"],
        E=["Spain", "Sweden", "Poland", "Slovakia"],
        F=["Hungary", "Portugal", "France", "Germany"]),
        EURO_R16_2020, THIRDS_2020, THIRDS_2020_WINNERS,
        venues={"R16": ["Spain", "England", "Romania", "Denmark",
                        "Scotland", "England", "Hungary", "Netherlands"],
                "QF": ["Germany", "Russia", "Italy", "Azerbaijan"],
                "SF": ["England", "England"], "final": ["England"]}),
    _euro16("Euro 2024", "2024-06-14", "2024-07-14", {"Germany"}, _g(
        A=["Germany", "Scotland", "Hungary", "Switzerland"],
        B=["Spain", "Croatia", "Italy", "Albania"],
        C=["Slovenia", "Denmark", "Serbia", "England"],
        D=["Poland", "Netherlands", "Austria", "France"],
        E=["Belgium", "Slovakia", "Romania", "Ukraine"],
        F=["Turkey", "Georgia", "Portugal", "Czech Republic"]),
        EURO_R16_2020, THIRDS_2020, THIRDS_2020_WINNERS),
]

BY_NAME = {f.name: f for f in [WC2026] + BACKTEST}


def matches(fmt, df):
    """All rows of this tournament, in date order."""
    return df[(df.tournament == fmt.tournament)
              & (df.date >= fmt.start) & (df.date <= fmt.end)]


def group_matches(fmt, df):
    """The group-stage fixtures: 6 per group, which precede every knockout match."""
    m = matches(fmt, df)
    return m.iloc[:6 * len(fmt.groups)]


def knockout_matches(fmt, df):
    """Knockout rows, in date order (after the group stage)."""
    m = matches(fmt, df)
    return m.iloc[6 * len(fmt.groups):]


# ---------------------------------------------------------------- ground truth
def match_winner(row, shootouts):
    """Winner of a knockout row. results.csv scores include extra time; a draw
    was settled on penalties and its winner is in shootouts.csv."""
    if row.home_score > row.away_score:
        return row.home_team
    if row.away_score > row.home_score:
        return row.away_team
    s = shootouts[(shootouts.date == row.date) & (shootouts.home_team == row.home_team)
                  & (shootouts.away_team == row.away_team)]
    assert len(s) == 1, f"no shootout for {row.date.date()} {row.home_team} v {row.away_team}"
    return s.winner.iloc[0]


def knockout_rounds(fmt, df):
    """The real knockout matches grouped by round, e.g. {"R16": rows, ..., "final": rows}.

    Rounds follow date order. A World Cup third-place playoff is dropped: both
    its teams already count as semi-finalists.
    """
    rows = list(knockout_matches(fmt, df).itertuples())
    has_playoff = fmt.tournament == "FIFA World Cup"
    assert len(rows) == 2 ** len(fmt.ko_rounds) - 1 + has_playoff, fmt.name
    out, i = {}, 0
    for k, rnd in enumerate(fmt.ko_rounds):
        size = 2 ** (len(fmt.ko_rounds) - 1 - k)
        if rnd == "final":
            out[rnd] = rows[-1:]
        else:
            out[rnd] = rows[i:i + size]
            i += size
    return out


def actual_stages(fmt, df, shootouts):
    """Stage each team reached, as an index into fmt.stages (0 = group exit)."""
    teams = {t for ts in fmt.groups.values() for t in ts}
    stage = dict.fromkeys(teams, 0)
    rounds = knockout_rounds(fmt, df)
    for k, rnd in enumerate(fmt.ko_rounds[:-1]):
        for r in rounds[rnd]:
            for t in (r.home_team, r.away_team):
                stage[t] = max(stage[t], k + 1)
    (final,) = rounds["final"]
    champion = match_winner(final, shootouts)
    runner_up = final.away_team if champion == final.home_team else final.home_team
    stage[runner_up], stage[champion] = len(fmt.ko_rounds), len(fmt.ko_rounds) + 1
    return stage
