# Live LLM Agent Research Outputs

## How To Read

- `01_per_scenario/<scenario>/tables/`: Portfolio Hall-compatible tables generated from each live LLM run.
- `02_cross_scenario_comparison/manager_scenario_comparison.csv`: cross-scenario manager results.
- `02_cross_scenario_comparison/live_scenario_social_summary.csv`: social activity and graph statistics.
- `03_figures/`: report-ready comparison figures.

## Best Manager By Scenario

| scenario | portfolio | total_return | sharpe | max_drawdown |
| --- | --- | --- | --- | --- |
| adversarial_persuader | drawdown_constrained_agent_portfolio | -0.0227928 | -1.39951 | -0.0515822 |
| barbell_two_camps | drawdown_constrained_agent_portfolio | 0.00394152 | 0.511204 | -0.0152264 |
| baseline_isolated | drawdown_constrained_agent_portfolio | -0.000554672 | -0.0439899 | -0.0297462 |
| bridge_broker | drawdown_constrained_agent_portfolio | -0.0336686 | -1.09089 | -0.0759958 |
| chain_influence | drawdown_constrained_agent_portfolio | -0.0230829 | -0.96181 | -0.0708079 |
| core_periphery | drawdown_constrained_agent_portfolio | -0.00765493 | -0.625031 | -0.0325525 |
| dense_market | hedge_agent_portfolio | 0.00081172 | 0.0281087 | -0.0744081 |
| echo_chambers | drawdown_constrained_agent_portfolio | -0.0142917 | -1.27638 | -0.0263325 |
| mentor_pairs | correlation_aware_agent_portfolio | -1.65007e-05 | -0.0141232 | -0.00332839 |
| star_social_hub | drawdown_constrained_agent_portfolio | -0.0178275 | -1.08881 | -0.0466471 |
| strategy_silos | equal_agent_portfolio | -0.0454547 | -2.08391 | -0.0649269 |
| truthful_anchor | drawdown_constrained_agent_portfolio | -0.0148788 | -1.1939 | -0.0344748 |

## Social Summary

| scenario | rounds | messages | trades | friend_requests | friend_accepts | final_friendships | avg_final_pnl |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adversarial_persuader | 50 | 1741 | 286 | 72 | 60 | 66 | -3851.86 |
| barbell_two_camps | 50 | 1727 | 473 | 58 | 56 | 66 | -47.7033 |
| baseline_isolated | 50 | 1694 | 454 | 74 | 66 | 66 | -1913.31 |
| bridge_broker | 50 | 1706 | 449 | 64 | 58 | 66 | -4247.29 |
| chain_influence | 50 | 1724 | 441 | 74 | 63 | 66 | -4761.5 |
| core_periphery | 50 | 1713 | 461 | 79 | 63 | 66 | -2723.86 |
| dense_market | 50 | 1703 | 498 | 0 | 0 | 66 | -291.347 |
| echo_chambers | 50 | 1730 | 468 | 65 | 61 | 66 | -3090.49 |
| mentor_pairs | 50 | 1741 | 438 | 65 | 59 | 66 | -6.52333 |
| star_social_hub | 50 | 1726 | 418 | 75 | 60 | 66 | -3081.57 |
| strategy_silos | 50 | 1725 | 458 | 64 | 60 | 66 | -4478.01 |
| truthful_anchor | 50 | 1719 | 448 | 64 | 60 | 66 | -3233.86 |