from mep_routing.installations.cli import CliRouteAnalysis


def test_cli_plot_score_uses_cli_length_bend_and_crossing_costs():
    routes = (("Supply Air Tree", (
        ((0, 0), (100, 0)),
        ((100, 0), (100, 100)),
    )),)

    score = CliRouteAnalysis(4000, 5).score(routes, crossings=2)

    assert score == 44_200
