# Data Structure and Information
---
The raw Expedia Hotel Search data records **what a user saw on a search results page** on Expedia.  
In the raw files ([available here](https://www.kaggle.com/competitions/expedia-personalized-sort/)), each time a user performs a search (`srch_id`), Expedia shows a **list of hotels**, and the raw dataset contains **one row per *property shown* in that list**.  
So at the raw level:
- **One search = one results page.**  
- **Each row = one hotel shown on that page.**  
- A single `srch_id` can therefore appear many times (because many hotels were shown).
---
This raw list-view dataset converted into a **longitudinal, time-indexed format**:
1. Each *search destination* (identified by `srch_destination_id`) is treated as its own **small MDP**.
2. Searches for the same destination are **sorted chronologically**, and we assign a **time index** (`time_idx = 1, 2, 3, ...`) representing the order in which searches happened at that destination.
3. For each search (`srch_id`), we **collapse all the property-level rows into a single row**, summarizing:
   - state features of the search,
   - action-like aggregate descriptors of the list,
   - reward outcomes such as revenue or clicks.

This produces a **long-format panel** where:
- **Each row = one search event**  
- **Each destination = one trajectory over time**  
- **`time_idx` = the search order for that destination**

This is the format required for downstream **longitudinal modeling and RL-style analysis**.
---

## MDP Structure Evaluation

The following are performed on the data to test the MDP setting:

1. Estimated lag-($k$) autocorrelations and test. (This tests for the Dynamic structure across time)
2. Compared models for $\mathbb{E}[R_t |  A_t]$ vs  $E[R_t | S_t, A_t]$. (compare Bandit vs MDP.)
3. Compared $E[S_{t+1} | S_t,A_t])$ vs  $E[S_{t+1} | S_t, A_t, S_{t-1}, A_{t-1}]$. (Markov property)
4. Used feature importance in $E[R_t |  S_t,A_t]$ to quantify the influence of each state/action dimension. 

From the result:

- Autocorrelations show weak but real temporal structure.
- Predicting reward improves when adding states.
- Predicting next states depends on previous states/actions.
- It is not strictly Markov, but somehow Markov. I think removing some variables we can restore strictly markov too.

  *We can safely conclude that this is **offline RL (MDP) problem**, not a **bandit**.*
---

## Data Dictionary , ID, State, Action, Reward


## Dataset Size Summary

| Stage / File                  | Description                                                | Rows     | Columns | Notes                                                                                   |
|------------------------------|------------------------------------------------------------|----------|---------|-----------------------------------------------------------------------------------------|
| List-level MDP: train        | Training split (destinations)                             | 14,879  | 26      | 496 distinct `srch_destination_id`                                                    |
| List-level MDP: test         | Test split                                                | 2,836   | 26      | 100 distinct `srch_destination_id`                                                    |

Each destination (`srch_destination_id`) forms a small longitudinal trajectory indexed by `time_idx = 1,2,…`, and splits are **group-pure**: no destination appears in more than one of train/test.

---

## Final List-Level MDP Data Dictionary (1 row per `srch_id`)

| Column Name              | Type      | Description                                                                                        |
|--------------------------|-----------|----------------------------------------------------------------------------------------------------|
| `srch_id`                | Integer   | Search/session ID; one row per unique search                                                       |
| `srch_destination_id`    | Integer   | Destination area ID; groups searches into longitudinal trajectories                                |
| `time_idx`               | Integer   | Order of the search within its destination, by earliest `date_time` (1, 2, …, K)                   |
| `srch_length_of_stay`    | Integer   | Number of nights searched for this query (LOS, truncated to ≤ 14)                                  |
| `srch_room_count`        | Integer   | Number of rooms requested in the search                                                            |
| `srch_saturday_night_bool` | Integer | 1 if the requested stay includes a Saturday night; 0 otherwise                                     |
| `random_bool`            | Integer   | 1 if the property list for this search was randomly ordered; 0 if “normal” Expedia ranking         |
| `prop_review_score`      | Float     | Representative (e.g., first) average review score of properties in the list (1–5)                  |
| `prop_location_score2`   | Float     | Representative secondary location desirability score for the list                                  |
| `prop_location_score1`   | Float     | Representative primary location desirability score for the list                                    |
| `prop_log_historical_price` | Float  | Representative log historical price for properties in the list (after imputation)                  |
| `prop_starrating`        | Integer   | Representative star rating of properties in the list                                               |
| `comp_rate`              | Integer   | 1 if **any** competitor is cheaper than Expedia in the list; 0 otherwise                           |
| `comp_inv`               | Integer   | 1 if **any** competitor is unavailable in the list; 0 otherwise                                    |
| `mean_hist_price`        | Float     | Mean of `prop_log_historical_price` across properties in the list                                  |
| `std_hist_price`         | Float     | Standard deviation of `prop_log_historical_price` across properties in the list                    |
| `corr_pos_price`         | Float     | Within-list Pearson correlation between `position` (rank in search results) and `price_usd`        |
| `corr_pos_review`        | Float     | Within-list Pearson correlation between `position` and `prop_review_score`                         |
| `n_props`                | Integer   | Number of distinct properties (`prop_id`) shown in this search                                     |
| `total_gross_revenue`    | Float     | Sum of `gross_bookings_usd` over all properties shown in this search                               |
| `gross_revenue_per_night`| Float     | `total_gross_revenue` divided by `srch_length_of_stay`; search-level revenue normalized by LOS     |
| `total_clicks`           | Integer   | Total count of `click_bool = 1` over all properties in the list                                    |
| `total_promotions`       | Integer   | Total number of properties with `promotion_flag = 1` in the list                                   |
| `avg_price_per_night`    | Float     | Mean `price_usd` across all properties shown in this search                                        |
| `std_price_usd`          | Float     | Standard deviation of `price_usd` across properties in the list                                    |
| `_split`                 | Categorical | Split label: `"train"`,  or `"test"`, based on destination-level group-pure partition |

Conceptually, `gross_revenue_per_night` and `total_clicks` are the main **reward signals**,  
while `total_promotions`, `avg_price_per_night`, `corr_pos_price`, and `corr_pos_review` summarize the **action-like list configuration**, and all remaining variables form the **state** of each time-indexed search.


