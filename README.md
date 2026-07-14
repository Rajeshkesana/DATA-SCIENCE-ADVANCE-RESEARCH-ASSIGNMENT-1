Reading Germany's Electricity Rhythm

This repository contains my Advanced Data Science assignment on forecasting electricity demand in Germany. My aim was not simply to fit several models and select the smallest error. I wanted to understand what each model learned, what information it was allowed to use, and why a complicated method did not always improve on a simple seasonal forecast.

The submitted work belongs to **Rajesh Kesana** and contains the executed notebook together with the final eight-page report.

- `Rajesh Kesana.ipynb` contains the complete analysis and saved outputs.
- `Rajesh Kesana.docx` contains my discussion of the results and answers to the assignment questions.

The repository is available at [DATA-SCIENCE-ADVANCE-RESEARCH-ASSIGNMENT-1](https://github.com/Rajeshkesana/DATA-SCIENCE-ADVANCE-RESEARCH-ASSIGNMENT-1).

## Where my investigation began

I used the German load series published by Open Power System Data. The selected variable was `DE_load_actual_entsoe_transparency`, covering every hour from 1 January 2015 to 30 September 2020. After checking the timestamps and numeric values, I retained 50,400 hourly observations. The series contained no missing load values.

I then produced two additional views of the same data: 2,100 daily averages and 301 Sunday-labelled weekly averages. The weekly version was useful for the statistical and regression models, while the original hourly version was kept for the LSTM. Berlin mean temperature from Open-Meteo and selected German public holidays were added later as possible explanatory information.

The source electricity file used by the notebook is:

```text
https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv
```

## What I noticed before modelling

The exploratory figures showed that German electricity use has a strong calendar rhythm. Demand changes by hour, working days behave differently from weekends, and the annual pattern remains visible after weekly aggregation. I inspected monthly movement, weekday profiles, decomposition, rolling behaviour and correlation structure rather than relying on a single line chart.

ADF and KPSS tests were considered together because the two tests begin from different null hypotheses. The level series was not treated as safely stationary. Differencing reduced the long-term movement, although seasonal dependence was still visible. This was also clear from the ACF and PACF plots and influenced the later SARIMA search.

## The simple forecasts were not throwaway models

I held back the final 104 weekly observations, which is approximately two years, and trained the weekly models on the earlier 197 observations. Four basic forecasts were tested first.

The mean forecast produced RMSE 4397.30 MW, MAE 3788.83 MW and MAPE 6.97%. The naive forecast produced RMSE 4459.11 MW, MAE 3783.20 MW and MAPE 6.79%. Drift was the weakest of the four, with RMSE 5117.96 MW, MAE 4339.89 MW and MAPE 8.05%.

Seasonal naive was much more convincing. By repeating the load from 52 weeks earlier, it achieved RMSE 3006.76 MW, MAE 2318.52 MW and MAPE 4.41%. This became the benchmark that the more advanced weekly models genuinely needed to beat. For me, this was one of the clearest findings in the assignment: annual memory carried more useful information than a general upward or downward trend.

## My SARIMA search and the lesson from AIC

The SARIMA stage covered every required combination of `p` from 0 to 6, `d` from 0 to 2 and `q` from 0 to 6. This gave 147 fitted orders. The seasonal part was held at `(1,1,1,52)` so that the comparison remained manageable and retained the yearly weekly cycle.

The lowest training AIC belonged to `(2,2,6)`, with AIC 1482.27. However, its two-year test RMSE was 9434.25 MW, so minimum AIC did not translate into a useful long-horizon forecast. I therefore used a separate 52-week validation section inside the training data to examine the strongest AIC candidates without selecting a model from the final test targets.

That process selected `(0,1,6)`. After refitting it on the full training section, its final errors were RMSE 3796.12 MW, MAE 3077.38 MW and MAPE 5.84%. Residual history, residual distribution and residual autocorrelation were also checked. The model was more defensible than choosing directly from the test set, but seasonal naive still performed better.

## When weather entered the forecast

For SARIMAX, I matched the weekly electricity values with Berlin temperature. I used the current weekly temperature and its one-week lag as exogenous variables. Both fitted temperature coefficients were negative, about -88.59 and -101.06, which agreed with the expectation that colder conditions are associated with greater electricity demand.

The SARIMAX result was RMSE 4347.12 MW, MAE 3515.29 MW and MAPE 6.69%. It did not improve on seasonal naive or the selected SARIMA model. I have treated this as a conditional forecast because observed weather from the test period was supplied. In a real future forecast, those values would need to come from a weather forecast rather than from observations that were already known afterwards.

## A weekly model that could learn nonlinear behaviour

I built Gradient Boosting features from earlier load values, shifted rolling summaries, annual and shorter lags, cyclical calendar terms, lagged temperature and holiday indicators. Shifting was important because a predictor for a particular week should not contain that week's target value.

The stored final model used 300 trees, learning rate 0.05, maximum depth 3 and subsample 0.8. It achieved RMSE 2397.84 MW, MAE 1808.56 MW and MAPE 3.45%. This was the strongest weekly modelling result in the main comparison.

I also carried out a separate feature ablation using learning rate 0.03. Its full feature set produced RMSE 2384.55 MW and MAPE 3.38%. That number belongs to the ablation experiment, so it should not be mistaken for the result of the stored final model. The importance plot showed that the 52-week load lag accounted for roughly 71.8% of fitted importance. Even here, the model depended mainly on the previous year's pattern.

## My hourly LSTM experiment

The LSTM used hourly observations and a scaler fitted only on the training period. I compared three small architectures using the final 10% of the training sequences as validation data. The two-layer model with a 336-hour look-back gave the lowest scaled validation RMSE, 0.0309, and was therefore retained.

On the hourly test observations, the final LSTM recorded RMSE 1642.27 MW, MAE 1271.52 MW and MAPE 2.28%. When its hourly predictions were averaged into weeks, the resulting diagnostic gave RMSE 693.18 MW, MAE 610.14 MW and MAPE 1.07%.

I do not interpret the weekly average as a completely separate 104-week-ahead model. The network makes rolling one-hour-ahead predictions while earlier observed test hours remain available as context. Averaging those predictions naturally smooths part of the hourly error. This is a different forecasting condition from asking SARIMA to produce the whole two-year weekly block at once.

## How I read the final outcome

There was no single winner under every possible forecasting condition. Seasonal naive was the best of the fixed elementary rules and even defeated both state-space models. Gradient Boosting was the best weekly feature-based result, but it operated with updated lag information. The LSTM had the lowest recorded errors, although it was evaluated through rolling hourly prediction rather than the same fixed weekly origin.

The practical conclusion I took from the experiment is that the forecasting protocol matters as much as the model name. A low error is meaningful only after stating the time scale, horizon and information available at the forecast origin. The results also suggest that Germany's yearly load pattern is extremely influential, because the 52-week lag remained important in both the simplest benchmark and the nonlinear regression model.

## Running my notebook

Google Colab is the easiest environment for reproducing the saved work. The full notebook is computationally demanding because the SARIMA section attempts 147 models and the LSTM is trained afterwards. Depending on the runtime, a complete execution may take several hours.

To reproduce it:

1. Open `Rajesh Kesana.ipynb` in Google Colab.
2. Make sure the runtime has internet access for OPSD and Open-Meteo.
3. Install any library that is unavailable in the selected runtime.
4. Run the cells from the beginning and keep their existing order.
5. Allow the SARIMA loop and the LSTM training to finish before comparing the final outputs.

For a local environment, the following commands provide the main packages:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install jupyter numpy pandas matplotlib requests holidays python-dateutil scikit-learn statsmodels tensorflow
jupyter notebook "Rajesh Kesana.ipynb"
```

Useful checkpoints are 50,400 hourly observations, 301 weekly observations, a 197/104 weekly split and 147 SARIMA candidates. If these values differ, the data period, aggregation rule or preprocessing order should be checked before trusting the later results.

## A final note on limitations

This study uses Berlin temperature as an approximate weather signal for an entire country. The test period also contains the unusual electricity behaviour of 2020, which was not represented in the earlier years in the same way. Gradient Boosting relies on realised earlier test weeks, and the LSTM relies on realised earlier test hours. Feature importance describes how the fitted trees used a variable; it does not prove that the variable caused electricity consumption to change.

These limitations do not make the comparison useless, but they set boundaries around the conclusions. A future extension could use weather from several German regions, a strict rolling-origin evaluation shared by all models, probabilistic intervals for machine-learning forecasts and explicit treatment of structural changes during 2020.

Electricity observations are credited to Open Power System Data, and historical temperature observations are credited to Open-Meteo. The research sources supporting the statistical and recurrent forecasting methods are listed in the submitted report.
